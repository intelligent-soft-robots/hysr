import context
from hysr import hysr_types
from hysr import observations


def test_pack():

    box = (0.0, 0.0, 0.0), (10.0, 10.0, 10.0)
    max_velocity = 1.0
    max_angular_velocity = 1.0
    
    main_sim = hysr_types.MainSimState (
        (5.,5.,5.),
        (0.5,0.5,0.5),
        (0.,0.,0.,0.),
        (0.5,0.5,0.5,0.5),
        (
            (2.25,2.25,2.25),
            (1.,0.,0.,0.,1.,0.,0.,0.,1.)
        ),
        context.ContactInformation(),
        -1,
        -1
    )

    packed = observations.pack_main_sim_state(
        main_sim,
        ("ball_position",),
        box, max_velocity, max_angular_velocity
    )

    assert tuple(packed) == (0.5,0.5,0.5)
