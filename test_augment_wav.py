import argparse

import torchaudio


from static_methods_realization import time_domain_inversion, \
                                random_phase_generation, \
                                high_frequency_addition, \
                                inaudible_voice_command, \
                                crossover, mutation, \
                                hidden_voice_commands, \
                                time_mask

from generative_models_usage import MelGanTool, DiffWaveTool


separator = '/'
def s(filename):
    return separator.join(filename.split('%'))


def test_augmentations(os: str):
    separator = '/' if os == 'linux' else '\\'

    source_audio_left, sample_rate_left = torchaudio.load(s('test%source_examples%left.wav'))
    source_audio_right, sample_rate_right = torchaudio.load(s('test%source_examples%right.wav'))
    '''
    # TimeDomainInversion
    tdi_audio_left = time_domain_inversion(source_audio_left)
    tdi_audio_right = time_domain_inversion(source_audio_right)

    torchaudio.save(s('test%augmented_examples%tdi_left.wav'), tdi_audio_left, sample_rate = sample_rate_left)
    torchaudio.save(s('test%augmented_examples%tdi_right.wav'), tdi_audio_right, sample_rate = sample_rate_right)
    
    # RandomPhaseGeneration
    rpg_audio_left = random_phase_generation(source_audio_left)
    rpg_audio_right = random_phase_generation(source_audio_right)

    torchaudio.save(s('test%augmented_examples%rpg_left.wav'), rpg_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%rpg_right.wav'), rpg_audio_right, sample_rate=sample_rate_right)

    # HighFrequencyAddition
    hfa_audio_left = high_frequency_addition(source_audio_left)
    hfa_audio_right = high_frequency_addition(source_audio_right)

    torchaudio.save(s('test%augmented_examples%hfa_left.wav'), hfa_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%hfa_right.wav'), hfa_audio_right, sample_rate=sample_rate_right)

    # InaudibleVoiceCommands
    ivc_audio_left = inaudible_voice_command(source_audio_left)
    ivc_audio_right = inaudible_voice_command(source_audio_right)

    torchaudio.save(s('test%augmented_examples%ivc_left.wav'), ivc_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%ivc_right.wav'), ivc_audio_right, sample_rate=sample_rate_right)

    #Crossover
    source_audio_left2, sample_rate_left = torchaudio.load(s('test%source_examples%left2.wav'))
    source_audio_right2, sample_rate_right = torchaudio.load(s('test%source_examples%right2.wav'))

    crossover_audio_left = crossover(source_audio_left, source_audio_left2)
    crossover_audio_right = crossover(source_audio_right, source_audio_right2)

    torchaudio.save(s('test%augmented_examples%crossover_left.wav'), crossover_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%crossover_right.wav'), crossover_audio_right, sample_rate=sample_rate_right)
    
    #Mutation
    mutation_audio_left = mutation(source_audio_left)
    mutation_audio_right = mutation(source_audio_right)

    torchaudio.save(s('test%augmented_examples%mutation_left.wav'), mutation_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%mutation_right.wav'), mutation_audio_right, sample_rate=sample_rate_right)

    #HiddenVoiceCommands
    hvc_audio_left = hidden_voice_commands(source_audio_left)
    hvc_audio_right = hidden_voice_commands(source_audio_right)

    torchaudio.save(s('test%augmented_examples%hvc_left.wav'), hvc_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%hvc_right.wav'), hvc_audio_right, sample_rate=sample_rate_right)

    melgan_tool = MelGanTool()

    melgan_audio_left = melgan_tool.augment_audio(source_audio_left)
    melgan_audio_right = melgan_tool.augment_audio(source_audio_right)

    torchaudio.save(s('test%augmented_examples%melgan_left.wav'), melgan_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%melgan_right.wav'), melgan_audio_right, sample_rate=sample_rate_right)
'''
    diffwave_tool = DiffWaveTool()

    diffwave_audio_left = diffwave_tool.augment_audio(source_audio_left)
    diffwave_audio_right = diffwave_tool.augment_audio(source_audio_right)

    torchaudio.save(s('test%augmented_examples%diffwave_left.wav'), diffwave_audio_left, sample_rate=sample_rate_left)
    torchaudio.save(s('test%augmented_examples%diffwave_right.wav'), diffwave_audio_right, sample_rate=sample_rate_right)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test augmentations method')
    parser.add_argument('os', help='specify the operating system on which the script is running',
                        choices=['linux', 'windows'])
    args = parser.parse_args()
    test_augmentations(args.os)
