"""Pluggable v2 approaches. Importing this package registers every approach in Matchups.V2.Base.REGISTRY."""

from Matchups.V2.Approaches import (KalmanTeam, CoachProfile, StintRapm, PairSynergy, OnOff, LineupFeatures, PlayTypeScheme,  # noqa: F401
                                    SizeSpeed, SwitchHunt, SchemeAssignment, ScheduleFatigue, ShotQuality,
                                    FourFactors, Clutch, TrackingPrior, LineupVsLineup, Stacker)
