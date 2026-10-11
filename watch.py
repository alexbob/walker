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


def main():
    if MODEL_PATH.exists():
        model = PPO.load(MODEL_PATH, device="cpu")
    else:
        model = None
        print("No trained Walker2d model yet. Showing an untrained two-leg preview.")
        print("Run python train.py to train it to walk.")

    env = make_env(render_mode=None, camera_id=-1, default_camera_config={"azimuth": 135, "elevation": -15, "distance": 4.0})

    try:
        obs, info = env.reset()
        # Render explicitly so simulation stops before any further render/reset.
        renderer = env.unwrapped.mujoco_renderer
        viewer = renderer._get_viewer("human")
        prevent_overlay_accumulation(viewer)
        exit_handler = ViewerExit(viewer)
        episode_reward = 0.0

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

            if terminated or truncated:
                print(f"Episode reward: {episode_reward:.1f}")
                episode_reward = 0.0
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
