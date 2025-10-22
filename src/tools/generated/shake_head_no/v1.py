from reachy_mini import ReachyMini
import time

def shake_head_no(robot: ReachyMini, times: int = 2, speed: float = 1.0) -> str:
    """
    Makes a robot shake its head to indicate 'no'.

    Args:
        robot: The ReachyMini robot object.
        times: The number of times to shake the head (default: 2).
        speed: The speed of the head movement (default: 1.0).

    Returns:
        A success message like "Successfully shook head 'no' X times".

    Raises:
        TypeError: If robot is not a ReachyMini object.
        ValueError: If times or speed are not positive.
    """

    if not isinstance(robot, ReachyMini):
        raise TypeError("robot must be a ReachyMini object")
    if times <= 0 or speed <= 0:
        raise ValueError("times and speed must be positive")

    try:
        for _ in range(times):
            robot.look_at_world(x=0, y=speed, z=0, duration=0.5)
            time.sleep(0.1)
            robot.look_at_world(x=0, y=-speed, z=0, duration=0.5)
            time.sleep(0.1)

        return f"Successfully shook head 'no' {times} times"
    except Exception as e:
        return f"Error shaking head: {e}"