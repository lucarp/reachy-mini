
from reachy_mini import ReachyMini
import time

def nod_yes(robot: ReachyMini, times: int = 2, speed: float = 1.0) -> str:
    """Make the robot nod 'yes' by moving its head up and down.

    This simulates a nodding motion by making the robot look at points
    at different heights in front of it.

    Args:
        robot: ReachyMini robot instance
        times: Number of times to nod (default: 2)
        speed: Speed multiplier for the movement (default: 1.0)

    Returns:
        Success message with number of nods performed

    Raises:
        ValueError: If times is not positive
        ValueError: If speed is not positive
    """
    # Input validation
    if times <= 0:
        raise ValueError("times must be positive")
    if speed <= 0:
        raise ValueError("speed must be positive")

    # Calculate duration based on speed
    duration = 0.5 / speed

    try:
        # Perform nodding motion by looking at different heights
        for i in range(times):
            # Look down (nod forward)
            robot.look_at_world(x=0.5, y=0.0, z=-0.2, duration=duration)
            time.sleep(duration)

            # Look up (nod back)
            robot.look_at_world(x=0.5, y=0.0, z=0.2, duration=duration)
            time.sleep(duration)

        # Return to neutral (look straight ahead)
        robot.look_at_world(x=0.5, y=0.0, z=0.0, duration=duration)
        time.sleep(duration)

        return f"Successfully nodded 'yes' {times} times"

    except Exception as e:
        return f"Error during nodding: {str(e)}"
