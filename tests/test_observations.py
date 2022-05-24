import pytest
import context
from hysr import hysr_types
from hysr import observations


def test_pack_main_sim_state():

    box = (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)
    max_velocity = 2.0
    max_angular_velocity = 1.0
    
    main_sim = hysr_types.MainSimState (
        (5.,7.5,5.),       # ball_position (0.5,0.75,0.5)
        (0.5,0.5,-1.0),     #  ball_velocity (0.625,0.625,0.25)
        (0.,0.,0.,0.),     # joint_positions (0.5,0.5,0.5,0.5)
        (0.5,0.5,0.5,0.5), # joint_velocities (0.75,0.75,0.75,0.75)
        (
            (2.5,2.5,2.5),               # racket_cartesian (position 3d) (0.25,0.25,0.25)
            (1.,0.,0.,0.,1.,0.,0.,0.,1.) # racket_cartesian (orientation) (1.,.5,.5,.5,1,.5,.5,.5,1)
        ),
        context.ContactInformation(), # contact (0,)
        -1, # iteration
        -1  # time_stamp
    )

    packed = observations.pack_main_sim_state(
        main_sim,
        ("ball_position",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.5,0.75,0.5)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("ball_velocity",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.625,0.625,0.25)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("ball_position","ball_velocity",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.5,0.75,0.5,0.625,0.625,0.25)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("ball_velocity","ball_position"),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.625,0.625,0.25,0.5,0.75,0.5)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("joint_positions",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.5,0.5,0.5,0.5)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("racket_cartesian",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.25,0.25,0.25,1.,.5,.5,.5,1,.5,.5,.5,1)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("racket_cartesian","contact"),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.25,0.25,0.25,1.,.5,.5,.5,1,.5,.5,.5,1,0.)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("joint_velocities",),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.75,0.75,0.75,0.75)

    packed = observations.pack_main_sim_state(
        main_sim,
        ("racket_cartesian","joint_velocities"),
        box, max_velocity, max_angular_velocity
    )
    assert tuple(packed) == (0.25,0.25,0.25,1.,.5,.5,.5,1,.5,.5,.5,1,0.75,0.75,0.75,0.75)

    with pytest.raises(ValueError):
        observations.pack_main_sim_state(
            main_sim,
            ("racket_cartesian","joint_velocities","iteration"),
            box, max_velocity, max_angular_velocity
        )

    with pytest.raises(ValueError):
        observations.pack_main_sim_state(
            main_sim,
            ("racket_cartesian","joint_velocities","time_stamp"),
            box, max_velocity, max_angular_velocity
        )

    def test_pack_pressure_robot_state():

        min_pressures = (
            (1000,1000) , 
            (2000,2000) ,
            (1000,2000) ,
            (1000,2000) ,
        )

        max_pressures = (
            (6000,6000) , 
            (6000,6000) ,
            (9000,6000) ,
            (6000,9000) ,
        )

        max_angular_velocity = 1.0

        state = hysr_types.PressureRobotState(
            (0.,math.pi,0.,-math.pi), # joint_positions (0.5,0.75,0.5,-0.75)
            (0.,1.,-1.,-0.5),         # joint_velocities (0.5,1.,-1.,0.25)
            ( # desired_pressures
                (1000,1000), # (0.,0.,1.,1.,0.25,0.5,0.75,0.75)
                (6000,6000), 
                (3000,4000),
                (4750,7250),
            ),
            ( # observed_pressures
                (1000,1000), # (0.,0.,1.,1.,0.25,0.5,0.5,0.25)
                (6000,6000), 
                (3000,4000),
                (3500,3750),
            ),
            -1, # iteration
            -1  # time_stamp
        )

        packed = observations.pack_pressure_robot_state(
            state,
            ("joint_positions",),
            min_pressures,
            max_pressures,
            max_angular_velocity
        )
        assert packed == (0.5,0.75,0.5,-0.75)

        packed = observations.pack_pressure_robot_state(
            state,
            ("desired_pressures",),
            min_pressures,
            max_pressures,
            max_angular_velocity
        )
        assert packed == (0.,0.,1.,1.,0.25,0.5,0.75,0.75)

        packed = observations.pack_pressure_robot_state(
            state,
            ("desired_pressures","observed_pressures"),
            min_pressures,
            max_pressures,
            max_angular_velocity
        )
        assert packed == (
            0.,0.,1.,1.,0.25,0.5,0.75,0.75,
            0.,0.,1.,1.,0.25,0.5,0.5,0.25
        )

        packed = observations.pack_pressure_robot_state(
            state,
            ("desired_pressures","joint_velocities","observed_pressures"),
            min_pressures,
            max_pressures,
            max_angular_velocity
        )
        assert packed == (
            0.,0.,1.,1.,0.25,0.5,0.75,0.75,
            0.5,1.,-1.,0.25,
            0.,0.,1.,1.,0.25,0.5,0.5,0.25
        )

        
