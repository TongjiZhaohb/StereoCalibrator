"""CUDA 向量化双旋转立体相机外参标定。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]
DeviceLike = Optional[Union[str, torch.device]]


def _to_numpy(value: ArrayLike) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _device(value: DeviceLike) -> torch.device:
    result = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    return result


def _validate(
    points_left: np.ndarray,
    points_right: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
) -> None:
    if points_left.ndim != 2 or points_left.shape[1] != 2:
        raise ValueError(f"points_left 应为 (N, 2)，实际为 {points_left.shape}")
    if points_right.shape != points_left.shape:
        raise ValueError("左右匹配点的形状必须相同")
    if K_left.shape != (3, 3) or K_right.shape != (3, 3):
        raise ValueError("相机内参必须为 (3, 3)")
    if not all(
        np.isfinite(value).all()
        for value in (points_left, points_right, K_left, K_right)
    ):
        raise ValueError("输入包含 NaN 或 Inf")


def _project_to_so3(matrix: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(matrix)
    U[:, -1] *= np.linalg.det(U @ Vt)
    return U @ Vt


def _split_pose(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    baseline = -np.asarray(t, dtype=np.float64).reshape(3)
    length = np.linalg.norm(baseline)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError("平移向量退化")
    x_axis = baseline / length
    reference = (
        np.array([0.0, 1.0, 0.0])
        if abs(x_axis[2]) > 0.95
        else np.array([0.0, 0.0, 1.0])
    )
    y_axis = np.cross(reference, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    R_right = _project_to_so3(np.stack((x_axis, y_axis, z_axis)))
    R_left = _project_to_so3(R_right @ R)
    return R_left, R_right


def _initial_rotations(
    points_left: np.ndarray,
    points_right: np.ndarray,
    K_left: np.ndarray,
    K_right: np.ndarray,
    ransac_method: int,
    threshold_px: float,
) -> Tuple[np.ndarray, np.ndarray]:
    identity = np.eye(3, dtype=np.float64)
    if len(points_left) < 5:
        return identity.copy(), identity.copy()

    left = (points_left - K_left[[0, 1], [2, 2]]) / K_left[
        [0, 1], [0, 1]
    ]
    right = (points_right - K_right[[0, 1], [2, 2]]) / K_right[
        [0, 1], [0, 1]
    ]
    mean_focal = np.mean(
        (K_left[0, 0], K_left[1, 1], K_right[0, 0], K_right[1, 1])
    )
    try:
        E, mask = cv2.findEssentialMat(
            left,
            right,
            identity,
            method=ransac_method,
            prob=0.99999,
            threshold=float(threshold_px) / max(float(mean_focal), 1e-12),
        )
        if E is None:
            return identity.copy(), identity.copy()

        best_pose = None
        best_count = -1
        for candidate in np.asarray(E).reshape(-1, 3, 3):
            count, R, t, _ = cv2.recoverPose(
                candidate, left, right, identity, 1e9, mask=mask
            )
            if count > best_count:
                best_count = int(count)
                best_pose = (R, t)
        return (
            _split_pose(*best_pose)
            if best_pose is not None
            else (identity.copy(), identity.copy())
        )
    except (cv2.error, ValueError, np.linalg.LinAlgError):
        return identity.copy(), identity.copy()


def _so3_exp(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), device=vector.device, dtype=vector.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    theta2 = torch.dot(vector, vector)
    theta = torch.sqrt(theta2)
    epsilon = 1e-8
    safe_theta = theta.clamp_min(epsilon)
    safe_theta2 = theta2.clamp_min(epsilon * epsilon)
    A = torch.where(
        theta < epsilon,
        1.0 - theta2 / 6.0,
        torch.sin(theta) / safe_theta,
    )
    B = torch.where(
        theta < epsilon,
        0.5 - theta2 / 24.0,
        (1.0 - torch.cos(theta)) / safe_theta2,
    )
    return torch.eye(3, device=vector.device, dtype=vector.dtype) + A * skew + B * (
        skew @ skew
    )


def _residuals(
    rays_left: torch.Tensor,
    rays_right: torch.Tensor,
    R_left: torch.Tensor,
    R_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = rays_left @ R_left.T
    right = rays_right @ R_right.T
    residual = torch.atan2(left[:, 1], left[:, 2]) - torch.atan2(
        right[:, 1], right[:, 2]
    )
    return residual, left, right


def _robust_cost(residual: torch.Tensor, delta: float) -> torch.Tensor:
    magnitude = residual.abs()
    return torch.where(
        magnitude <= delta,
        residual.square(),
        2.0 * delta * magnitude - delta * delta,
    ).sum()


def _cost(
    rays_left: torch.Tensor,
    rays_right: torch.Tensor,
    R_left: torch.Tensor,
    R_right: torch.Tensor,
    robust_delta: float,
    regularization_weight: float,
) -> torch.Tensor:
    residual, _, _ = _residuals(rays_left, rays_right, R_left, R_right)
    result = _robust_cost(residual, robust_delta)
    if regularization_weight:
        result = result + regularization_weight * R_right[1, 2].square()
    return result


def _linearize(
    rays_left: torch.Tensor,
    rays_right: torch.Tensor,
    R_left: torch.Tensor,
    R_right: torch.Tensor,
    robust_delta: float,
    regularization_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, left, right = _residuals(
        rays_left, rays_right, R_left, R_right
    )
    epsilon = torch.finfo(rays_left.dtype).eps
    xl, yl, zl = left.unbind(dim=1)
    xr, yr, zr = right.unbind(dim=1)
    denominator_left = (yl.square() + zl.square()).clamp_min(epsilon)
    denominator_right = (yr.square() + zr.square()).clamp_min(epsilon)

    # atan2(y, z) 对左乘 SO(3) 扰动的解析雅可比。
    J_left = torch.stack(
        (
            -torch.ones_like(xl),
            xl * yl / denominator_left,
            xl * zl / denominator_left,
        ),
        dim=1,
    )
    J_right = torch.stack(
        (
            torch.ones_like(xr),
            -xr * yr / denominator_right,
            -xr * zr / denominator_right,
        ),
        dim=1,
    )
    J = torch.cat((J_left, J_right), dim=1)

    magnitude = residual.abs()
    weights = torch.where(
        magnitude <= robust_delta,
        torch.ones_like(residual),
        robust_delta / magnitude.clamp_min(epsilon),
    )
    H = J.T @ (J * weights[:, None])
    b = -(J.T @ (weights * residual))
    cost = _robust_cost(residual, robust_delta)

    if regularization_weight:
        axis = R_right[:, 2]
        J_regularizer = torch.stack(
            (
                axis.new_zeros(()),
                axis.new_zeros(()),
                axis.new_zeros(()),
                -axis[2],
                axis.new_zeros(()),
                axis[0],
            )
        )
        H = H + regularization_weight * torch.outer(
            J_regularizer, J_regularizer
        )
        b = b - regularization_weight * J_regularizer * R_right[1, 2]
        cost = cost + regularization_weight * R_right[1, 2].square()
    return cost, H, b


@torch.no_grad()
def optimize_birotation(
    rays_left: torch.Tensor,
    rays_right: torch.Tensor,
    R_left: torch.Tensor,
    R_right: torch.Tensor,
    *,
    max_iterations: int = 100,
    mu: float = 1e-3,
    robust_delta: float = 1e-3,
    regularization_weight: float = 1.0,
    tolerance: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """在 SO(3)×SO(3) 上求解双旋转。"""
    if (
        rays_left.ndim != 2
        or rays_left.shape[1] != 3
        or rays_right.shape != rays_left.shape
    ):
        raise ValueError("左右射线必须是形状相同的 (N, 3) 张量")
    if not len(rays_left):
        return R_left, R_right, rays_left.new_zeros(())

    identity6 = torch.eye(6, device=rays_left.device, dtype=rays_left.dtype)
    damping = float(mu)
    previous_cost = float("inf")
    final_cost = rays_left.new_tensor(float("inf"))

    for _ in range(max_iterations):
        cost, H, b = _linearize(
            rays_left,
            rays_right,
            R_left,
            R_right,
            robust_delta,
            regularization_weight,
        )
        system = H + damping * identity6
        try:
            update = torch.linalg.solve(system, b)
        except RuntimeError:
            update = torch.linalg.pinv(system) @ b

        candidate_left = _so3_exp(update[:3]) @ R_left
        candidate_right = _so3_exp(update[3:]) @ R_right
        candidate_cost = _cost(
            rays_left,
            rays_right,
            candidate_left,
            candidate_right,
            robust_delta,
            regularization_weight,
        )
        cost_value, candidate_value, update_norm = torch.stack(
            (cost, candidate_cost, torch.linalg.vector_norm(update))
        ).cpu().tolist()

        if np.isfinite(candidate_value) and candidate_value <= cost_value:
            R_left, R_right = candidate_left, candidate_right
            final_cost = candidate_cost
            damping = max(damping * 0.5, 1e-9)
            relative_change = abs(previous_cost - candidate_value) / max(
                abs(previous_cost), 1e-12
            )
            previous_cost = candidate_value
            if relative_change < tolerance or update_norm < tolerance:
                break
        else:
            final_cost = cost
            damping = min(damping * 2.0, 1e9)

    return R_left, R_right, final_cost


def calibrate_stereo(
    points_left: ArrayLike,
    points_right: ArrayLike,
    K_left: ArrayLike,
    K_right: ArrayLike,
    *,
    baseline: float = 1.0,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.float32,
    max_iterations: int = 100,
    mu: float = 1e-3,
    robust_delta: float = 1e-3,
    regularization_weight: float = 1.0,
    tolerance: float = 1e-6,
    ransac_method: int = cv2.RANSAC,
    ransac_threshold_px: float = 0.5,
    use_essential_initialization: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """由二维匹配点估计右相机相对于左相机的旋转和平移。

    在线立体标定默认从单位旋转开始，以避免 CPU RANSAC 开销。相机初始偏差
    较大时可启用 ``use_essential_initialization`` 扩大收敛范围。
    """
    points_left_np = np.asarray(_to_numpy(points_left), dtype=np.float64)
    points_right_np = np.asarray(_to_numpy(points_right), dtype=np.float64)
    K_left_np = np.asarray(_to_numpy(K_left), dtype=np.float64)
    K_right_np = np.asarray(_to_numpy(K_right), dtype=np.float64)
    _validate(points_left_np, points_right_np, K_left_np, K_right_np)
    if not len(points_left_np):
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    if use_essential_initialization:
        initial_left, initial_right = _initial_rotations(
            points_left_np,
            points_right_np,
            K_left_np,
            K_right_np,
            ransac_method,
            ransac_threshold_px,
        )
    else:
        initial_left = np.eye(3, dtype=np.float64)
        initial_right = np.eye(3, dtype=np.float64)
    homogeneous_left = np.column_stack(
        (points_left_np, np.ones(len(points_left_np)))
    )
    homogeneous_right = np.column_stack(
        (points_right_np, np.ones(len(points_right_np)))
    )
    rays_left_np = np.linalg.solve(K_left_np, homogeneous_left.T).T
    rays_right_np = np.linalg.solve(K_right_np, homogeneous_right.T).T

    target = _device(device)
    rays_left = torch.as_tensor(rays_left_np, device=target, dtype=dtype)
    rays_right = torch.as_tensor(rays_right_np, device=target, dtype=dtype)
    R_left = torch.as_tensor(initial_left, device=target, dtype=dtype)
    R_right = torch.as_tensor(initial_right, device=target, dtype=dtype)
    R_left, R_right, _ = optimize_birotation(
        rays_left,
        rays_right,
        R_left,
        R_right,
        max_iterations=max_iterations,
        mu=mu,
        robust_delta=robust_delta,
        regularization_weight=regularization_weight,
        tolerance=tolerance,
    )

    relative_rotation = R_right.T @ R_left
    left = rays_left @ R_left.T
    right = rays_right @ R_right.T
    epsilon = torch.finfo(dtype).eps
    left_z = torch.where(
        left[:, 2].abs() < epsilon,
        torch.copysign(left[:, 2].new_full((), epsilon), left[:, 2]),
        left[:, 2],
    )
    right_z = torch.where(
        right[:, 2].abs() < epsilon,
        torch.copysign(right[:, 2].new_full((), epsilon), right[:, 2]),
        right[:, 2],
    )
    direction_sign = torch.where(
        (left[:, 0] / left_z).mean() > (right[:, 0] / right_z).mean(),
        -1.0,
        1.0,
    )
    translation = float(baseline) * direction_sign * R_right[0]
    return (
        relative_rotation.cpu().numpy().astype(np.float64, copy=False),
        translation.cpu().numpy().astype(np.float64, copy=False),
    )


__all__ = ["calibrate_stereo", "optimize_birotation"]
