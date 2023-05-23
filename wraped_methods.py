import pandas as pd
import torch
import random
from typing import List

from tqdm import tqdm

from static_methods_realization import time_domain_inversion, \
                                random_phase_generation, \
                                high_frequency_addition, \
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
    def __init__(self, min_window: int = 10, max_window: int = 10, p: float = None):
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
    def __init__(self,
                 min_raising_frequency: int = 32,
                 max_raising_frequency: int = 32,
                 p: float = None):
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
    def __init__(self, acceptable_n_ftt_list: List[int], p: float = None):
        super().__init__(p)
        self.n_ftt_list = acceptable_n_ftt_list
    def get_n_ftt(self):
        return random.choice(self.n_ftt_list)
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            n_ftt = self.get_n_ftt()
            return random_phase_generation(samples, n_ftt)
        return samples

class Crossover(BaseAugmentation):
    def __init__(self,
                 min_header_length = 100,
                 max_header_length = 1000,
                 min_skip_step = 2,
                 max_skip_step = 8,
                 p: float = None):
        super().__init__(p)
        self.header_length = (min_header_length, max_header_length)
        self.skip_step = (min_skip_step, max_skip_step)

    def get_header_length(self):
        return random.randint(self.header_length[0], self.header_length[1])

    def get_skip_step(self):
        return random.randint(self.skip_step[0], self.skip_step[1])
    def apply_on_df(self, df: pd.DataFrame):
        labels_dict = sorted(list(set(df['label'])))
        splitted_data = dict([(i, []) for i in labels_dict])
        for index, row in df.iterrows():
            splitted_data[row["label"]].append(row["wave"])
        augmented_data = []

        for label, waves in splitted_data.items():
            for _ in tqdm(range(len(waves))):
                num_to_select = 2
                cross_pair = random.sample(waves, num_to_select)
                audio_new = torch.Tensor(cross_pair[0]) if random.random() < 0.5 \
                    else torch.Tensor(cross_pair[1])

                if self.get_p():
                    header_length = self.get_header_length()
                    skip_step = self.get_skip_step()
                    audio_new = crossover(torch.Tensor(cross_pair[0]),
                                          torch.Tensor(cross_pair[1]),
                                          header_length=header_length,
                                          skip_step=skip_step)
                    augmented_data.append({"wave": audio_new.cpu(),
                                           "label": label})
                else:
                    augmented_data.append(audio_new)

        return pd.DataFrame(augmented_data)

    def apply_on_pair(self, samples_x1: torch.Tensor, samples_x2: torch.Tensor):
        if self.get_p():
            header_length = self.get_header_length()
            skip_step = self.get_skip_step()
            return crossover(samples_x1, samples_x2, header_length, skip_step)
        return random.choice([samples_x1, samples_x2])

#inaudible_voice_command, mutation, hidden_voice_commands
class Mutation(BaseAugmentation):
    def __init__(self, min_header_length: int = 100, max_header_length: int = 1000,
                       min_mutation_p: float = 0.0001, max_mutation_p: float = 0.0005,
                       p: float = None):
        super().__init__(p)
        self.data_max = 32767
        self.data_min = -32768
        self.mutation_p = (min_mutation_p, max_mutation_p)
        self.header_length = (min_header_length, max_header_length)
    def get_header_length(self):
        return random.randint(self.header_length[0], self.header_length[1])
    def get_mutation_p(self):
        return random.uniform(self.mutation_p[0], self.mutation_p[1])
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            mutation_p = self.get_mutation_p()
            header_length = self.get_header_length()
            return mutation(samples,
                            header_length=header_length,
                            mutation_p=mutation_p)
        return samples

class HiddenVoiceCommands(BaseAugmentation):
    def __init__(self, in_n_ftt: List[int], out_hop_length: List[int], p: float = None):
        super().__init__(p)
        self.in_n_ftt_list = in_n_ftt
        self.out_hop_length_list = out_hop_length
    def get_in_n_ftt(self):
        return random.choice(self.in_n_ftt_list)
    def get_out_hop_length(self):
        return random.choice(self.out_hop_length_list)
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            in_n_ftt = self.get_in_n_ftt()
            out_hop_length = self.get_out_hop_length()
            return hidden_voice_commands(samples, in_n_ftt, out_hop_length)
        return samples

class MelGan(BaseAugmentation):
    def __init__(self, min_gauss: float = 0.5, max_gauss: float = 2, p: float = None):
        super().__init__(p)
        self.melgan = MelGanTool()
        self.gauss = (min_gauss, max_gauss)
    def get_gauss(self):
        return random.uniform(self.gauss[0], self.gauss[1])
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            gauss_coeff =  self.get_gauss()
            return self.melgan.augment_audio(samples, gauss_coeff)
        return samples

class DiffWave(BaseAugmentation):
    def __init__(self, min_gauss: float = 0.5, max_gauss: float = 2, p: float = None):
        super().__init__(p)
        self.diffwave = DiffWaveTool()
        self.gauss = (min_gauss, max_gauss)
    def get_gauss(self):
        return random.uniform(self.gauss[0], self.gauss[1])
    def apply(self, samples: torch.Tensor):
        if self.get_p():
            gauss_coeff =  self.get_gauss()
            return self.diffwave.augment_audio(samples, gauss_coeff)
        return samples


if __name__ == '__main__':
    pass


