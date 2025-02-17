# https://github.com/willisma/SiT


from src.modules.transport.transport import ModelType, PathType, Transport, WeightType


class CreateTransport:

    def __init__(
        self,
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
    ):
        self.path_type = path_type
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def __call__(self):
        """Function for creating Transport object **Note**: model prediction defaults to velocity.

        Args:
        - path_type: type of path to use; default to linear
        - learn_score: set model prediction to score
        - learn_noise: set model prediction to noise
        - velocity_weighted: weight loss by velocity weight
        - likelihood_weighted: weight loss by likelihood weight
        - train_eps: small epsilon for avoiding instability during training
        - sample_eps: small epsilon for avoiding instability during sampling
        """

        if self.prediction == "noise":
            model_type = ModelType.NOISE
        elif self.prediction == "score":
            model_type = ModelType.SCORE
        elif self.prediction == "data":
            model_type = ModelType.DATA
        else:
            model_type = ModelType.VELOCITY

        if self.loss_weight == "velocity":
            loss_type = WeightType.VELOCITY
        elif self.loss_weight == "likelihood":
            loss_type = WeightType.LIKELIHOOD
        else:
            loss_type = WeightType.NONE

        path_choice = {
            "Linear": PathType.LINEAR,
            "GVP": PathType.GVP,
            "VP": PathType.VP,
        }

        path_type = path_choice[self.path_type]

        if path_type in [PathType.VP]:
            train_eps = 1e-5 if self.train_eps is None else self.train_eps
            sample_eps = 1e-3 if self.sample_eps is None else self.sample_eps
        elif path_type in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
            train_eps = 1e-3 if self.train_eps is None else self.train_eps
            sample_eps = 1e-3 if self.sample_eps is None else self.sample_eps
        else:  # velocity & [GVP, LINEAR] is stable everywhere
            train_eps = 0
            sample_eps = 0

        # create flow state
        state = Transport(
            model_type=model_type,
            path_type=path_type,
            loss_type=loss_type,
            train_eps=train_eps,
            sample_eps=sample_eps,
        )

        return state
