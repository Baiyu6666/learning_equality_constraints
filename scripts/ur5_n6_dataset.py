from __future__ import annotations

import numpy as np

try:
    from ur5_pybullet_utils import UR5PyBulletKinematics
except Exception:
    from scripts.ur5_pybullet_utils import UR5PyBulletKinematics


def sample_ur5_upward_dataset(
    n_train: int,
    n_grid: int,
    seed: int,
    urdf_path: str,
    ee_link_index: int | None,
    tool_axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    ur5 = UR5PyBulletKinematics.from_settings(
        urdf_path=str(urdf_path),
        ee_link_index=ee_link_index,
        tool_axis=str(tool_axis),
    )
    lo = ur5.joint_lower.reshape(1, -1)
    hi = ur5.joint_upper.reshape(1, -1)
    lo1 = ur5.joint_lower.astype(np.float32)
    hi1 = ur5.joint_upper.astype(np.float32)
    try:
        import pybullet as p  # type: ignore
    except Exception as e:
        raise RuntimeError(f"pybullet unavailable in sampling: {e}")

    link_ids = sorted(set([int(v) for v in ur5.arm_joint_indices] + [int(ur5.ee_link_index)]))
    check_pairs: list[tuple[int, int]] = []
    for i in range(len(link_ids)):
        for j in range(i + 1, len(link_ids)):
            a = int(link_ids[i])
            b = int(link_ids[j])
            # Adjacent kinematic neighbors are always very close; skip them.
            if abs(a - b) <= 1:
                continue
            check_pairs.append((a, b))

    def _axis(q: np.ndarray) -> np.ndarray:
        return ur5.tool_axis_batch(q.reshape(1, 6).astype(np.float32))[0].astype(np.float32)

    def _res(axis: np.ndarray) -> float:
        return float(np.linalg.norm(axis[:2]))

    def _self_collision(q: np.ndarray, margin: float = 0.0) -> bool:
        ur5._set_q(q.astype(np.float32))
        for a, b in check_pairs:
            pts = p.getClosestPoints(
                bodyA=ur5.robot_id,
                bodyB=ur5.robot_id,
                distance=float(margin),
                linkIndexA=int(a),
                linkIndexB=int(b),
                physicsClientId=ur5.client_id,
            )
            if len(pts) > 0:
                return True
        return False

    def _solve_wrist_up(q4: np.ndarray, n_restart: int = 4, iters: int = 18) -> np.ndarray | None:
        # Solve q5,q6 such that tool axis aligns with +z.
        best_q = None
        best_res = 1e18
        for _ in range(max(1, int(n_restart))):
            q = np.zeros((6,), dtype=np.float32)
            q[:4] = q4.astype(np.float32)
            q[4] = rng.uniform(lo1[4], hi1[4])
            q[5] = rng.uniform(lo1[5], hi1[5])
            for _it in range(max(1, int(iters))):
                a = _axis(q)
                r = np.array([a[0], a[1]], dtype=np.float32)
                val = float(np.linalg.norm(r))
                if a[2] > 0.0 and val < best_res:
                    best_res = val
                    best_q = q.copy()
                if a[2] > 0.0 and val < 0.03:
                    return q.copy()

                eps = 2e-3
                J = np.zeros((2, 2), dtype=np.float32)
                qd = q.copy()
                qd[4] = np.clip(qd[4] + eps, lo1[4], hi1[4])
                ad = _axis(qd)
                J[:, 0] = (ad[:2] - r) / eps
                qd = q.copy()
                qd[5] = np.clip(qd[5] + eps, lo1[5], hi1[5])
                ad = _axis(qd)
                J[:, 1] = (ad[:2] - r) / eps

                JTJ = J.T @ J + 1e-4 * np.eye(2, dtype=np.float32)
                step = np.linalg.solve(JTJ, J.T @ r)
                step = np.clip(step, -0.25, 0.25)
                q[4] = np.clip(q[4] - step[0], lo1[4], hi1[4])
                q[5] = np.clip(q[5] - step[1], lo1[5], hi1[5])
        if best_q is not None and best_res < 0.06:
            return best_q
        return None

    def _build(n: int) -> np.ndarray:
        out = []
        tries = 0
        max_tries = max(300, 30 * n)
        print(f"[data][ur5] target={n}, max_tries={max_tries}, mode=solve(q5,q6|q1..q4)")
        while len(out) < n and tries < max_tries:
            tries += 1
            q4 = rng.uniform(lo1[:4], hi1[:4], size=(4,)).astype(np.float32)
            q = _solve_wrist_up(q4, n_restart=4, iters=18)
            if q is None:
                continue
            a = _axis(q)
            if (a[2] > 0.0) and (_res(a) < 0.06) and (not _self_collision(q, margin=0.0)):
                out.append(q.astype(np.float32))
            if tries == 1 or (tries % 20 == 0):
                print(f"[data][ur5] tries={tries}, accepted={len(out)}/{n}")
        if not out:
            raise RuntimeError("unable to sample UR5 upward-orientation manifold points by wrist solve")
        arr = np.stack(out, axis=0).astype(np.float32)
        if len(arr) >= n:
            idx = rng.choice(len(arr), size=n, replace=False)
            return arr[idx].astype(np.float32)
        idx = rng.choice(len(arr), size=n, replace=True)
        return arr[idx].astype(np.float32)

    x_train = _build(max(1, int(n_train)))
    grid = _build(max(1, int(n_grid)))
    return x_train, grid


def sample_ur5_upward_dataset_analytic(n_train: int, n_grid: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Pure-Python sampler: no pybullet, no collision checking.
    # Orientation convention matches demo analytic chain: axes z,y,y,y,x,z; tool axis local +x.
    rng = np.random.default_rng(int(seed))
    lo = np.array([-np.pi] * 6, dtype=np.float32)
    hi = np.array([np.pi] * 6, dtype=np.float32)

    def _axis(q: np.ndarray) -> np.ndarray:
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        R = np.eye(3, dtype=np.float32)
        axes = ("z", "y", "y", "y", "x", "z")
        for j, ax in enumerate(axes):
            a = float(q[j])
            if ax == "x":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
            elif ax == "y":
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
            else:
                c, s = np.cos(a), np.sin(a)
                Rj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
            R = R @ Rj
        a = R @ ex
        return a / max(float(np.linalg.norm(a)), 1e-8)

    def _solve_wrist_up(q4: np.ndarray, n_restart: int = 4, iters: int = 16) -> np.ndarray | None:
        best_q = None
        best_res = 1e18
        for _ in range(max(1, int(n_restart))):
            q = np.zeros((6,), dtype=np.float32)
            q[:4] = q4.astype(np.float32)
            q[4] = rng.uniform(lo[4], hi[4])
            q[5] = rng.uniform(lo[5], hi[5])
            for _it in range(max(1, int(iters))):
                a = _axis(q)
                r = np.array([a[0], a[1]], dtype=np.float32)
                val = float(np.linalg.norm(r))
                if (a[2] > 0.0) and (val < best_res):
                    best_res = val
                    best_q = q.copy()
                if (a[2] > 0.0) and (val < 0.03):
                    return q.copy()
                eps = 2e-3
                J = np.zeros((2, 2), dtype=np.float32)
                qd = q.copy()
                qd[4] += eps
                ad = _axis(qd)
                J[:, 0] = (ad[:2] - r) / eps
                qd = q.copy()
                qd[5] += eps
                ad = _axis(qd)
                J[:, 1] = (ad[:2] - r) / eps
                JTJ = J.T @ J + 1e-4 * np.eye(2, dtype=np.float32)
                step = np.linalg.solve(JTJ, J.T @ r)
                step = np.clip(step, -0.25, 0.25)
                q[4] = np.clip(q[4] - step[0], lo[4], hi[4])
                q[5] = np.clip(q[5] - step[1], lo[5], hi[5])
        if best_q is not None and best_res < 0.06:
            return best_q
        return None

    def _build(n: int) -> np.ndarray:
        out = []
        tries = 0
        max_tries = max(300, 30 * n)
        print(f"[data][ur5_py] target={n}, max_tries={max_tries}")
        while len(out) < n and tries < max_tries:
            tries += 1
            q4 = rng.uniform(lo[:4], hi[:4], size=(4,)).astype(np.float32)
            q = _solve_wrist_up(q4, n_restart=4, iters=16)
            if q is None:
                continue
            a = _axis(q)
            if (a[2] > 0.0) and (np.linalg.norm(a[:2]) < 0.06):
                out.append(q.astype(np.float32))
            if tries == 1 or (tries % 20 == 0):
                print(f"[data][ur5_py] tries={tries}, accepted={len(out)}/{n}")
        if not out:
            raise RuntimeError("unable to sample analytic UR5-up dataset")
        arr = np.stack(out, axis=0).astype(np.float32)
        if len(arr) >= n:
            idx = rng.choice(len(arr), size=n, replace=False)
        else:
            idx = rng.choice(len(arr), size=n, replace=True)
        return arr[idx].astype(np.float32)

    x_train = _build(max(1, int(n_train)))
    grid = _build(max(1, int(n_grid)))
    return x_train, grid
