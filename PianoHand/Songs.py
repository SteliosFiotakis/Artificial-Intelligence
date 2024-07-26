import numpy as np
import asyncio
import mujoco


async def godfather_waltz(data: mujoco.MjData, left_hand: slice, right_hand: slice):
    """
    Todo: add description for the songs
    :param data: Data instance of mujoco.MjData
    :param left_hand: Slice object for indexing left hand ctrl
    :param right_hand: Slice object for indexing right hand ctrl
    """

    #########################################
    # INITIALIZATION OF TIME RELATED THINGS #
    #########################################

    beat = 0.5  # 90 BPM / 1.5 BPS / 0.67 SPB (B=Beats, P=Per, M=Minutes, S=Seconds)

    whole = 4 * beat
    half = 2 * beat
    quarter = beat
    eighth = beat / 2
    sixteenth = beat / 4

    ####################################
    # INITIALIZATION OF HAND POSITIONS #
    ####################################

    # Left Hand
    d2_f = np.asarray([-0.3, -0.25, -0.3, 0., 1.4, 0.05, 0.05, 1.4, 0., 0.,
                       0., 0., -0.05, -0.05, 1.4, 0., 0.2, 0., 0.1, 0.2,
                       -0.205, 0.11, -0.055, ])
    d2_p = np.asarray([-0.3, -0.25, -0.3, 0., 1.4, 0.05, 0.05, 1.4, 0., 0.,
                       0., 0., -0.05, 0.2, 1.4, 0., 0.2, 0., 0.1, 0.2,
                       -0.205, 0.11, -0.055, ])

    dfa3_f = np.asarray([-0.3, -0.25, -0.3, 0., 1.4, 0.05, 0.05, 1.4, 0., 0.,
                         0., 0., 0., 0., 0., 0., 0.2, 0., 0.1, 0.2,
                         -0.335, 0.11, -0.055, ])
    dfa3_p = np.asarray([-0.3, -0.25, -0.3, 0.15, 1.4, 0.05, 0.2, 1.4, 0., 0.,
                         0., 0., 0., 0., 0., 0., 0.35, 0., 0.1, 0.2,
                         -0.335, 0.11, -0.08, ])

    # Right Hand
    main1_init = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, 0.3, 0.45,
                             0.5, 0., 0.1, 0.45, 0.55, 0., 0.45, 0., -0.35, -0.15,
                             -0.07, 0.155, -0.015, ])
    main1_c5 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.5, 0.3, 0.45,
                           0.5, 0., 0.1, 0.45, 0.55, 0., 0.45, 0., -0.35, -0.15,
                           -0.07, 0.155, -0.015, ])
    main1_b4 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.5, 0.2, 1.4, 0.95, 0.25, 0.45,
                           0.5, 0., 0.1, 0.45, 0.55, 0., 0.55, 0., -0.35, -0.15,
                           -0.07, 0.155, -0.015, ])
    # main1_csharp5 = np.asarray([-0.2 ,  -0.25 ,  0.15 ,  1.4 ,   0.5 ,   0.2,    1.4 ,   0.95,   0.3 ,   0.45,
    #                              0.5 ,   0.   ,  0.1  ,  0.45,   0.55,   0. ,    0.45,   0.  ,  -0.35,  -0.15,
    #                             -0.07,   0.155, -0.015,])
    main1_csharp5 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, 0.3, 0.8,
                                0.5, 0., 0.1, 0.45, 0.55, 0., 0.45, 0., -0.35, -0.15,
                                -0.07, 0.155, -0.015, ])
    # main1_gsharp4 = np.asarray([-0.2 ,  -0.25 ,  0.15 ,  1.4 ,   0.95,   0.2,    1.4 ,   0.95,   0.3 ,   0.8 ,
    #                              0.5 ,   0.   ,  0.1  ,  0.45,   0.55,   0. ,    0.45,   0.  ,  -0.35,  -0.15,
    #                             -0.07,   0.155, -0.015,])
    main1_gsharp4 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, 0.3, 0.4,
                                0.5, 0., 0.1, 0.45, 0.55, 0., 0.8, 0., -0.35, -0.15,
                                -0.07, 0.155, -0.015, ])
    main1_dsharp5 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, 0.3, 0.45,
                                0.5, 0., 0.1, 0.7, 0.55, 0., 0.45, 0., -0.35, -0.15,
                                -0.07, 0.155, -0.015, ])
    main3_d5 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, -0.05, 0.9,
                           0.5, 0., 0.1, 0.45, 0.55, 0., 0.45, 0., -0.35, -0.15,
                           -0.07, 0.155, -0.015, ])

    main2_init = np.asarray([-0.2, -0.1, 0.2, 1.4, 0.9, 0.2, 1.4, 0.9, 0.3, 0.45,
                             0.15, 0., 0.1, 0.35, 0.2, -0.1, 0.4, 0., -0.05, -0.5,
                             -0.075, 0.155, 0., ])
    main2_c5 = np.asarray([-0.2, -0.1, 0.2, 1.4, 0.9, 0.2, 1.4, 0.4, 0.3, 0.45,
                           0.15, 0., 0.1, 0.35, 0.2, -0.1, 0.4, 0., -0.05, -0.5,
                           -0.075, 0.155, 0., ])
    main2_g4 = np.asarray([-0.2, -0.1, 0.2, 1.4, 0.9, 0.2, 1.4, 0.9, 0.3, 0.45,
                           0.15, 0., 0.1, 0.35, 0.2, -0.1, 0.8, 0., -0.05, -0.5,
                           -0.07, 0.155, 0., ])

    main3_b4 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.5, 0.2, 1.4, 0.95, 0.05, 0.65,
                           0.5, 0., 0.1, 0.45, 0.55, 0., 0.55, 0., -0.35, -0.15,
                           -0.07, 0.155, -0.015, ])
    main3_g4 = np.asarray([-0.2, -0.25, 0.15, 1.4, 0.95, 0.2, 1.4, 0.95, 0.3, 0.45,
                           0.5, 0., 0.1, 0.45, 0.55, 0., 0.85, 0., 0.15, -0.15,
                           -0.07, 0.155, -0.015, ])

    ############
    # SEQUENCE #
    ############

    await asyncio.sleep(1)
    data.ctrl[right_hand] = main1_init
    await asyncio.sleep(1)

    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(whole)
    data.ctrl[right_hand] = main1_b4
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_csharp5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main1_gsharp4
    await asyncio.sleep(whole)

    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main1_b4
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_csharp5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main2_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main2_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main2_c5
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main2_g4
    await asyncio.sleep(whole)

    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main1_b4
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_dsharp5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main3_b4
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main3_d5
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(whole)

    data.ctrl[right_hand] = main1_gsharp4
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main3_g4
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_init
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = main1_gsharp4
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main1_c5
    await asyncio.sleep(half)
    data.ctrl[right_hand] = main2_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main3_b4
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = main2_g4
    await asyncio.sleep(whole)


async def moonlight_sonata(data: mujoco.MjData, left_hand: slice, right_hand: slice):
    """
       Todo: add description for the songs
       :param data: Data instance of mujoco.MjData
       :param left_hand: Slice object for indexing left hand ctrl
       :param right_hand: Slice object for indexing right hand ctrl
       """

    #########################################
    # INITIALIZATION OF TIME RELATED THINGS #
    #########################################

    beat = 0.4  # 150 BPM / 2.5 BPS / 0.4 SPB (B=Beats, P=Per, M=Minutes, S=Seconds)

    whole = 4 * beat
    half = 2 * beat
    quarter = beat
    eighth = beat / 2
    sixteenth = beat / 4

    ####################################
    # INITIALIZATION OF HAND POSITIONS #
    ####################################

    # Left Hand
    csharp2_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.24, 0.17, 0., ])
    csharp2_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.24, 0.17, -0.07, ])
    b1_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                       -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.175, 0.12, -0.02, ])
    b1_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                       -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.175, 0.12, -0.08, ])
    a1_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                       -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.15, 0.12, -0.02, ])
    a1_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                       -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.15, 0.12, -0.08, ])
    fsharp1_trans = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                                -0.3, 0.4, 0.3, 0., 0.3, -0.1, -0.4, 0.7, -0.11, 0.05, 0.1, ])
    fsharp1_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.11, 0.17, 0., ])
    fsharp1_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.11, 0.17, -0.07, ])
    gsharp1_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.14, 0.17, 0., ])
    gsharp1_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.14, 0.17, -0.07, ])
    csharp1_f = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.04, 0.17, 0., ])
    csharp1_p = np.asarray([-0.3, -0.3, 0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0.1,
                            -0.3, 0.4, 0.5, 0., 0.3, -0.1, -0.4, 0.7, -0.04, 0.17, -0.07, ])

    # Right Hand
    gcsharpe_init = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.15, 0.5, 0.4, 0.2, 0.65, 0.3, 0.,
                                0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.02, ])
    gcsharpe = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.15, 0.5, 0.4, 0.2, 0.65, 0.3, 0.,
                           0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
    gcsharpe_g = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.1, 0.5, 0.4, 0.2, 0.7, 0.3, 0.,
                             0.1, 0.4, 0.4, 0., 0.6, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
    gcsharpe_c = np.asarray([-0.3, -0.3, 0.25, 0.7, 0.5, 0.1, 0.5, 0.4, 0.2, 0.7, 0.3, 0.,
                             0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
    gcsharpe_e = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.1, 0.5, 0.4, 0.2, 0.9, 0.3, 0.,
                             0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])

    gce_trans = np.asarray([-0.3, -0.3, 0.1, 0.8, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                            -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gce = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                      -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gce_g = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                        -0.05, 0.5, 0.2, -0.2, 0.6, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gce_c = np.asarray([-0.3, -0.3, 0.1, 0.6, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                        -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gce_e = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.7, 0.7, 0.,
                        -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gdf_d = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.8, 0.4, -0.05, 0.4, 0.7, 0.,
                        -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gdf_d_trans = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.8, 0.4, -0.05, 0.4, 0.7, 0.,
                              -0.05, 0.5, 0.2, -0.2, 0.3, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
    gdf_f = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                        -0.05, 0.8, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])

    gcdef_trans = np.asarray([-0.3, -0.3, 0.3, 0.5, 0.5, 0.15, 0.75, 0.5, 0., 0.5, 0.5, 0.,
                              -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_g1 = np.asarray([-0.3, -0.3, 0.3, 0.5, 0.5, 0.15, 0.25, 0.5, 0., 0.5, 0.5, 0.,
                           -0.1, 0.2, 0.6, -0.1, 0.55, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_c1 = np.asarray([-0.3, -0.3, 0.3, 0.7, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                           -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_f = np.asarray([-0.3, -0.3, 0.3, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                          -0.1, 0.5, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_g2 = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                           -0.1, 0.2, 0.6, -0.1, 0.5, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_c_sharp1 = np.asarray([-0.3, -0.3, 0.1, 0.6, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                                 -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_e = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.7, 0.5, 0.,
                          -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    gcdef_g3 = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                           -0.1, 0.2, 0.6, -0.1, 0.5, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
    # gcdef_c_sharp2 = np.asarray([-0.3,  -0.3,   0.1,   0.6,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
    # gcdef_d_sharp1 = np.asarray([-0.3,  -0.3,   0.3,   0.3,   0.5,   -0.05,  0.55,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
    # gcdef_g4 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                        -0.1,   0.2,   0.6,  -0.1,   0.5,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
    # gcdef_c2 = np.asarray([-0.3,  -0.3,   0.3,   0.7,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                        -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
    # gcdef_d_sharp2 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,   -0.05,  0.55,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])

    #
    # gcdef = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,  -0.05,  0.4,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                     -0.1,   0.2,   0.6,  -0.1,   0.3,   0.  ,  0. ,  -0.4,   0.09,  0.15, -0.04,])
    # gcdef_g2 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,  -0.05,  0.4,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
    #                     -0.1,   0.2,   0.6,  -0.1,   0.3,   0.  ,  0. ,  -0.4,   0.09,  0.15, -0.04,])
    gcdef_c_sharp2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                                 5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                                 0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                                 -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                                 2.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                                 9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
    gcdef_d_sharp1 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 3.00000000e-01,
                                 5.00000000e-01, -5.00000000e-02, 5.50000000e-01, 5.00000000e-01,
                                 0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                                 -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                                 3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                                 9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
    gcdef_g4 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                           5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                           0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                           -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                           6.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                           9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
    gcdef_c2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 7.00000000e-01,
                           5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                           0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                           -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                           3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                           9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
    gcdef_d_sharp2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                                 5.00000000e-01, -5.00000000e-02, 5.50000000e-01, 5.00000000e-01,
                                 0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                                 -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                                 3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                                 9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])

    dfsharpc_init = np.asarray([-0.2, -0.3, 0.3, 0.2, 0.5, -0.05, 0.15, 0.5, 0., 0.1, 0.5, 0.,
                                0.05, 0.1, 0.1, -0.6, 0.1, 0., -0.5, -0.2, 0.15, 0.16, -0.02, ])
    dfsharpc = np.asarray([-0.2, -0.3, 0.3, 0.2, 0.5, -0.05, 0.15, 0.5, 0., 0.1, 0.5, 0.,
                           0.1, 0.3, 0.6, -0.6, 0.1, 0., -0.5, -0.2, 0.15, 0.16, -0.04, ])
    dfsharpc_d = np.asarray([-2.00000000e-01, -3.00000000e-01, 3.00000000e-01, 2.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 1.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 1.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -5.00000000e-02, 3.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             4.00000000e-01, 0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                             1.50000000e-01, 1.60000000e-01, -4.00000000e-02, ])
    dfsharpc_f = np.asarray([-2.00000000e-01, -3.00000000e-01, 3.00000000e-01, 6.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 1.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 1.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -5.00000000e-02, 3.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             1.00000000e-01, 0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                             1.50000000e-01, 1.60000000e-01, -4.00000000e-02, ])
    dfsharpc_c = np.asarray([-2.00000000e-01, -3.00000000e-01, 3.00000000e-01, 3.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 1.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 1.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -5.00000000e-02, 6.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             1.00000000e-01, 0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                             1.50000000e-01, 1.60000000e-01, -4.00000000e-02, ])

    end = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                      5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                      0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                      -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                      3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                      9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])

    ############
    # SEQUENCE #
    ############

    await asyncio.sleep(1)
    data.ctrl[right_hand] = gcsharpe_init
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(1)
    data.ctrl[right_hand] = gcsharpe
    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(sixteenth)

    # First Position (First Part)
    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_trans
    data.ctrl[left_hand] = a1_f
    await asyncio.sleep(quarter)

    # Second Position (First Part)
    data.ctrl[right_hand] = gce_g
    data.ctrl[left_hand] = a1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    data.ctrl[left_hand] = fsharp1_f
    await asyncio.sleep(quarter)

    # Third-Second Position (First Part)
    data.ctrl[right_hand] = gce_g
    data.ctrl[left_hand] = fsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d_trans
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    # Last Position (First Part)
    data.ctrl[right_hand] = gcdef_trans
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gcdef_g1
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c1
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g2
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c_sharp1
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g3
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c_sharp2
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_d_sharp1
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g4
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c2
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_d_sharp2
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = dfsharpc_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = dfsharpc
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_f
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_c
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_d
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    await asyncio.sleep(1)
    data.ctrl[right_hand] = end


async def melisses(data: mujoco.MjData, left_hand: slice, right_hand: slice):
    """
    Todo: add description for the songs
    :param data: Data instance of mujoco.MjData
    :param left_hand: Slice object for indexing left hand ctrl
    :param right_hand: Slice object for indexing right hand ctrl
    """

    ######################################
    # INITIALIZATION TIME RELATED THINGS #
    ######################################

    beat = 0.4  # 150 BPM / 2.5 BPS / 0.4 SPB (B=Beats, P=Per, M=Minutes, S=Seconds)

    whole = 4 * beat
    half = 2 * beat
    quarter = beat
    eighth = beat / 2
    sixteenth = beat / 4

    ####################################
    # INITIALIZATION OF HAND POSITIONS #
    ####################################

    fga_init = np.asarray([0., 0.05, -0.1, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                           1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                           -0.035, 0.145, 0.015, ])
    fga_f = np.asarray([0., 0.05, -0.1, 0.5, 0.5, 0., 1.15, 1., 0., 1.05,
                        1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                        -0.035, 0.145, 0.015, ])
    fga_g = np.asarray([0., 0.05, -0.1, 0.1, 0.5, 0., 1.15, 0.5, 0., 1.05,
                        1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                        -0.035, 0.145, 0.015, ])
    fga_a = np.asarray([0., 0.05, -0.1, 0.1, 0.5, 0., 1.15, 0.95, 0., 1.05,
                        0.5, 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                        -0.035, 0.145, 0.015, ])
    fga_sharp = np.asarray([0., 0.05, -0.1, 0.1, 0.5, 0., 1.15, 0.95, 0., 1.05,
                            1., 0., 0.1, 0.45, 0.6, 0., 0.6, 0., 0., 0.,
                            -0.035, 0.145, 0.015, ])
    fga_d = np.asarray([0., 0.05, -0.1, 0.1, 0.5, 0., 1.15, 0.5, 0., 1.05,
                        1., 0., 0.1, 0.05, 0.6, 0., 0.8, 0., 0., 0.,
                        -0.035, 0.145, 0.015, ])

    gacd_init = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                            1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                            -0.105, 0.145, 0.015, ])
    gacd_g = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                         1., 0., 0.1, 0.05, 0.6, 0., 0.8, 0., 0., -0.1,
                         -0.105, 0.145, 0.015, ])
    gacd_prea = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                            1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., -0.45, 0.,
                            -0.105, 0.145, 0.015, ])
    gacd_a = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                         1., 0., 0.1, 0.05, 0.6, 0., 0.85, 0., -0.45, 0.,
                         -0.105, 0.145, 0.015, ])
    gacd_asharp = np.asarray([0., 0.05, 0.15, 0.45, 0.6, 0., 1.15, 1., 0., 1.05,
                              1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., -0.45, 0.,
                              -0.105, 0.145, 0.015, ])
    gacd_c = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 0.5, 0., 1.05,
                         1., 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                         -0.105, 0.145, 0.015, ])
    gacd_d = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                         0.5, 0., 0.1, 0.05, 0.6, 0., 0.6, 0., 0., 0.,
                         -0.105, 0.145, 0.015, ])
    gacd_dsharp = np.asarray([0., 0.05, 0.15, 0.1, 0.6, 0., 1.15, 1., 0., 1.05,
                              1., 0., 0.1, 0.5, 0.6, 0., 0.6, 0., 0., 0.,
                              -0.105, 0.145, 0.015, ])

    ############
    # SEQUENCE #
    ############

    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(whole)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_dsharp
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_dsharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_g
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_g
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_a
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_dsharp
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_dsharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_init
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_asharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_a
    await asyncio.sleep(half)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = gacd_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gacd_c
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_sharp
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = fga_d
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_a
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_g
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = fga_init
    await asyncio.sleep(whole)
