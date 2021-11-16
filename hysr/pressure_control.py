import pam_mujoco
import typing
from .robot_state import RobotState, Pressures, Joints
from .pseudo_real_robot import PseudoRealRobot
from .real_robot import RealRobot
from .extra_balls import ExtraBallsSim
from .ball import BallSim


REAL_OR_PSEUDO_REAL = typing.Union[RealRobot,PseudoRealRobot]
SIM = typing.Union[BallSim,ExtraBallsSim]


class PressureControl:

    def __init__(self,
                 robot : REAL_OR_PSEUDO_REAL,
                 simulations : typing.Sequence[SIM]):

        self._robot : REAL_OR_PSEUDO_REAL = robot
        self._simulations : SIM = simulations


    def align(self,step:float=0.01,bursts_per_step:int=10)->None:
        """
        The (pseudo) real robot and the robot of the simulations are expected to 
        be in sync (i.e same joint positions and velocities). Yet, sometimes they 
        are not (e.g. at startup). Directly settings the robot positions and velocities
        to the simulations may "break" them (likely: mujoco does not support teleportation
        well). This function interpolate the simulated robot joints and velocities from
        their current values to the one of the (pseudo) real robot values.
        This functions calls the bursting functions of the simulations as appropriate.
        """
        
        # reading the target state (i.e. state of the (pseudo) real robot)
        robot_state : RobotState = self._robot.read()

        def _one_step(arg : typing.Tuple[Joints,Joints]) -> Typing[bool,Joints]:
            """
            Argument:
              arg (Joints,Joints): target joint value and current joint value.
            Returns:
              finished (bool): True if the current is close to the target
              new desired state (Joints): if finished is False, returns
                current+step or current-step, so that the difference
                between current and target decreases.
            """
            target, current = arg
            diff = target - current
            if abs(diff) < step:
                current = target
                return True, current
            else:
                if diff > 0:
                    current += step
                else:
                    current -= step
                return False, current

        def _align(target_positions : Joints, target_velocities : Joints,
                   simulation: SIM, bursts_per_step: INT):

            """
            Movies the joints of the robot of the simulation to the 
            target_positions and the target_velocities by interpolating from 
            their current state to the desired state
            """
            
            joints_state : JointsState = simulation.read_robot()

            over = [False] * len(target_positions)

            while not all(over):

                p = list(map(_one_step, zip(target_positions, joints_state.positions)))
                v = list(map(_one_step, zip(target_velocities, joints_State.velocities)))

                positions = [p_[1] for p_ in p]
                velocities = [v_[1] for v_ in v]

                over = [p_[0] for p_ in p]

                simulation.set_robot(positions, velocities, nb_iterations=1, burst=bursts_per_step)

        if len(self._simulations) == 1:
            _align(target_positions, target_velocities, self._simulations[0], bursts_per_step)

        else:
            threads = [
                threading.Thread(
                    target=_align,
                    args=(target_positions, target_velocities, simulation, bursts_per_step),
                )
                for simulation in self._simulations
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        
