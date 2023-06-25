

from python_utils_aisu.utils import Cooldown, CooldownVarU
from AnimationStates.animation import ChangerCycle
from AnimationStates.dynamics import SecondOrderDynamics
from AnimationsTha import animations

def getKwargs():
    return  {
		'animations': {
			'sentiment_happy_tha': {'weight': 1.0},
			'sentiment_surprised_tha': {'weight': 1.0},
            
			'idle_breathing_sin_tha': {},
			'idle_blinks_random_tha': {},
            
			'idle_eye_glance_random_tha': {},
            # TODO: change to unfocused after no speech event for a while
			# 'idle_eye_unfocused_random_tha': {},
            
			'idle_body_still_tha': {},
			'idle_body_sway_tha': {},
			'idle_body_head_sway_tha': {},
			'idle_body_head_bob_tha': {},
		},

		'transitions': {
			"by_type": {
				"default": ({'name': 'linear'}, {'seconds': 0}),
				"idle": ({'name': 'linear'}, {'seconds': 4}),
			},
			"None": {
				"default": ({'name': 'linear'}, {'seconds': 0}),
			},
		},
		'sentiments_args': {
			"surprise": {
				'duration_multiplier': 0.7,
				'transition_multiplier': 0.5,
			},
			"fear": {
				'duration_multiplier': 0.7,
				'transition_multiplier': 0.5,
			},
		},
		'state_changes': {
			'idle_body': ChangerCycle(CooldownVarU(8, variance=4)),
		},
        'dynamics': {
			# 'mouth': SecondOrderDynamics(
			# 	f=3,
			# 	z=0.7,
			# 	r=2.35,
			# ),
			# 'mouth': SecondOrderDynamics(
			# 	f=2.35,
			# 	z=0.46,
			# 	r=2.35,
			# ),
			# 'mouth': SecondOrderDynamics(
			# 	f=2.35,
			# 	z=0.46,
			# 	r=1.3,
			# ),
			# 'mouth': SecondOrderDynamics(
			# 	f=1.5,
			# 	z=0.55,
			# 	r=2.75,
			# ),
			'mouth': SecondOrderDynamics(
				# f=1.5,
				# z=0.55,
				# r=0.25,
				# f=2.5,
				# z=2.50,
				# r=1.0,
				f=2.5,
				z=0.40,
				r=1.0,
			),
		},
	}
