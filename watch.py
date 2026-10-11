from walking_env import MODEL_PATH, make_env
from stable_baselines3 import PPO
import glfw


class ViewerExit:
    """Request shutdown without destroying GLFW during an event callback."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.requested = False
        self.original_key_callback = viewer._key_callback
        glfw.set_key_callback(viewer.window, self.on_key)
        glfw.set_window_close_callback(viewer.window, self.on_close)

    def request(self):
        self.requested = True
        self.viewer._paused = False
        self.viewer._loop_count = 0

    def on_key(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE:
            if action == glfw.RELEASE:
                self.request()
            return
        self.original_key_callback(window, key, scancode, action, mods)

    def on_close(self, window):
        # Gymnasium otherwise destroys the window inside its render loop.
        glfw.set_window_should_close(window, False)
        self.request()


def prevent_overlay_accumulation(viewer):
    """Rebuild overlays for each redraw, including redraws while paused.

    Installed Gymnasium appends text on each paused redraw but only clears
    overlays after render() returns. Keep this workaround local to our viewer.
    """
    original_create_overlay = viewer._create_overlay

    def create_overlay():
        viewer._overlays.clear()
        original_create_overlay()

    viewer._create_overlay = create_overlay


def episode_summary(reward, steps, dt, start_x, end_x, terminated, truncated):
    duration = steps * dt
    distance = end_x - start_x
    speed = distance / duration if duration else 0.0
    reasons = []
    if terminated:
        reasons.append("fell / unhealthy posture")
    if truncated:
        reasons.append("time limit")
    reason = " + ".join(reasons) or "interrupted"
    return (f"Episode reward: {reward:.1f} | duration: {duration:.2f} s | "
            f"forward distance: {distance:.2f} m | average speed: {speed:.2f} m/s | "
            f"ended: {reason}")


def run_viewer(make_env, model_path, label, train_script):
    if model_path.exists():
        model = PPO.load(model_path, device="cpu")
    else:
        model = None
        print(f"No trained {label} model yet. Showing a zero-action preview.")
        print(f"Run python {train_script} to train it to walk.")

    env = make_env(render_mode=None, camera_id=-1, default_camera_config={"azimuth": 135, "elevation": -15, "distance": 4.0})

    try:
        obs, info = env.reset()
        # Render explicitly so simulation stops before any further render/reset.
        renderer = env.unwrapped.mujoco_renderer
        viewer = renderer._get_viewer("human")
        prevent_overlay_accumulation(viewer)
        exit_handler = ViewerExit(viewer)
        episode_reward = 0.0
        episode_steps = 0
        start_x = float(info["x_position"])

        while not exit_handler.requested:
            renderer.render("human")
            if exit_handler.requested:
                break
            if model is None:
                action = env.action_space.low * 0.0
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1

            if terminated or truncated:
                print(episode_summary(episode_reward, episode_steps, env.unwrapped.dt,
                                      start_x, float(info["x_position"]), terminated, truncated))
                episode_reward = 0.0
                obs, info = env.reset()
                episode_steps = 0
                start_x = float(info["x_position"])

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def main():
    run_viewer(make_env, MODEL_PATH, "Walker2d", "train.py")


if __name__ == "__main__":
    main()
