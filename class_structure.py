import torch
import random
from typing import List

from augmentation_utils import pad


from static_methods_realization import time_domain_inversion, \
                                random_phase_generation, \
                                high_frequency_addition, \
                                inaudible_voice_command, \
                                crossover, mutation, \
                                hidden_voice_commands


from generative_models_usage import MelGanTool, DiffWaveTool

class BaseAugmentation:
    def __init__(self, p: float = 0.5):
        self.p = p
    def get_p(self):
        return random.random() > self.p
    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        pass

class Compose:
    def __init__(self, augmentation_list: List[BaseAugmentation]):
        self.augmentation_list = augmentation_list
    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        augmented_samples = samples
        for augmentation in self.augmentation_list:
            augmented_samples = augmentation.apply(augmented_samples)
        return augmented_samples


class TimeDomainInversion(BaseAugmentation):
    def __init__(self, p, min_window: int = 10, max_window: int = 10):
        super().__init__(p)
        self.window = (min_window, max_window)
    def get_window(self):
        return random.randint(self.window[0], self.window[1])
    def apply(self, samples: torch.Tensor) -> torch.Tensor:
        if self.get_p():
            window = self.get_window()
            return time_domain_inversion(samples, window)
        return samples

class HighFrequencyAddition(BaseAugmentation):
    def __init__(self, p, n_fft, min_raising_frequency: int = 32, max_raising_frequency: int = 32):
        super().__init__(p)
        self.raising_frequency = (min_raising_frequency, max_raising_frequency)

    def get_frequency_value(self):
        return random.randint(self.raising_frequency[0], self.raising_frequency[1])

    def apply(self, samples: torch.Tensor):
        if self.get_p():
            frequency_value = self.get_frequency_value()
            return high_frequency_addition(samples,
                                           n_additional_stft=frequency_value,
                                           raising_frequency=frequency_value)
        return samples

class RandomPhaseGeneration(BaseAugmentation):
    def __init__(self, p, acceptable_n_ftt_list: List[int]):
        super().__init__(p)
        self.n_ftt_list = acceptable_n_ftt_list
    def get_n_ftt(self):
        return random.choice(self.n_ftt_list)
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            n_ftt = self.get_n_ftt()
            return random_phase_generation(samples, n_ftt)






if __name__ == '__main__':
    pass


