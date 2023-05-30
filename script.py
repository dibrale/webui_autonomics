import yaml
import math
from os import listdir, makedirs
from os.path import isfile, join, isdir

import gradio as gr

import modules.shared as shared
import numpy as np
from transformers import pipeline

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None, device="cpu")

params = {
    'print_debug': True,
    'temp_lo': 0.4,
    'temp_hi': 1.1,
    'typical_p_lo': 0.8,
    'typical_p_hi': 0.2,
    'repetition_penalty_lo': 1.15,
    'repetition_penalty_hi': 1.1,
    'encoder_repetition_penalty_lo': 1.05,
    'encoder_repetition_penalty_hi': 1,
    'penalty_alpha_lo': 2.5,
    'penalty_alpha_hi': 1,
    'top_k_lo': 4,
    'top_k_hi': 10,
}


def print_d(text):
    if params['print_debug']:
        print('[Autonomic System Extension] ' + text)


def autonomic_map(dict_list):
    out = []
    sentiment = [0, 0, 0, 0, 0, 0]
    print_d('--------------')
    print_d('Raw Sentiment Output')
    print_d('--------------')
    for i, d in enumerate(dict_list):
        print_d(f"{d['label']}: {d['score']}")
    print_d('--------------')
    print_d('Sentiment Component Contributions')
    print_d('--------------')
    # Anger
    # anger = sigmoid(15 * sentiment[0] - 10) + 0.2 * np.sin(math.pi*sentiment[0])**2

    # Anger
    sentiment[0] = [d['score'] for d in dict_list if d['label'] == 'anger'][0]
    anger = sentiment[0] ** 3
    out.append(anger)
    print_d(f'Anger: {np.round(anger, 3)}')

    # Disgust
    sentiment[1] = [d['score'] for d in dict_list if d['label'] == 'disgust'][0]
    disgust = sentiment[1]
    out.append(disgust)
    print_d(f'Disgust: {np.round(disgust, 3)}')

    # Fear
    sentiment[2] = [d['score'] for d in dict_list if d['label'] == 'fear'][0]
    fear = sentiment[2] ** 2
    out.append(fear)
    print_d(f'Fear: {np.round(fear, 3)}')

    # Joy
    sentiment[3] = [d['score'] for d in dict_list if d['label'] == 'joy'][0]
    joy = 0.75 * (sentiment[3] + 0.1) * np.cos(math.pi * sentiment[3]) ** 2
    out.append(joy)
    print_d(f'Joy: {np.round(joy, 3)}')

    # Neutral is the 4th value

    # Sadness
    sentiment[4] = [d['score'] for d in dict_list if d['label'] == 'sadness'][0]
    sadness = 0.75 * (sentiment[4] + 0.1) * np.cos(math.pi * sentiment[4]) ** 2
    out.append(sadness)
    print_d(f'Sadness: {np.round(sadness, 3)}')

    # Surprise
    sentiment[5] = [d['score'] for d in dict_list if d['label'] == 'surprise'][0]
    surprise = sentiment[5]
    out.append(surprise)
    print_d(f'Surprise: {np.round(surprise, 3)}')

    # out = anger + disgust + fear + joy + sadness + surprise

    # Normalize if desired
    final = np.sum(out)
    # final = np.sum(out) / np.sum(sentiment)

    # Ceiling
    if final > 1:
        final = 1

    print_d('--------------')
    print_d(f'Autonomic coefficient: {np.round(final, 2)}')
    print_d('--------------')

    return final


def range_bias(val: list[float | int, float | int], bias: float) -> float:
    out = val[0] + bias * (val[1] - val[0])
    return out


def make_parameters(bias, toggle):

    # TODO: Make this procedural
    params_new = {
        'temp': float(range_bias([params['temp_lo'], params['temp_hi']], bias)),
        'typical_p': float(range_bias([params['typical_p_lo'], params['typical_p_hi']], bias)),
        'repetition_penalty': float(range_bias([params['repetition_penalty_lo'], params['repetition_penalty_hi']], bias)),
        'encoder_repetition_penalty': float(range_bias([params['encoder_repetition_penalty_lo'], params['encoder_repetition_penalty_hi']], bias)),
        'penalty_alpha': float(range_bias([params['penalty_alpha_lo'], params['penalty_alpha_hi']], bias)),
        'top_k': int(range_bias([int(params['top_k_lo']), int(params['top_k_hi'])], bias))
    }

    if toggle == 0:
        file_name = "Autonomic_Buffer_A"
    else:
        file_name = "Autonomic_Buffer_B"

    with open(f'presets/{file_name}.yaml', 'w') as f:
        yaml.dump(params_new, f)


# Toggle function that uses an integer because the toggle has to be within Gradio
def which_params(toggle):
    if toggle == 0:
        buffer = "Autonomic_Buffer_A"
        toggle = 1
    else:
        buffer = "Autonomic_Buffer_B"
        toggle = 0

    return buffer, toggle


def autonomic_update(text, buffer):
    emotions = classifier(text)[0]
    make_parameters(autonomic_map(emotions), buffer)


def save_params(filename):
    with open(f'param_ranges/{filename}.yaml', 'w') as f:
        yaml.dump(params, f)
        print_d('--------------')
        print_d(f'Parameter ranges saved to \'param_ranges/{filename}.yaml\':')
        print_d('--------------')
        for k, v in params.items():
            print_d(f"{k}: {v:.2f}")


def load_params(filename):
    with open(f'param_ranges/{filename}.yaml') as f:
        params.update(yaml.safe_load(f))
        print_d('--------------')
        print_d(f'Parameter ranges after loading \'param_ranges/{filename}.yaml\':')
        print_d('--------------')
        for k, v in params.items():
            print_d(f"{k}: {v:.2f}")


def list_files(path):
    if not isdir(path):
        makedirs(path)
    files_list = [f for f in listdir(path) if isfile(join(path, f))]
    out_list = []
    suffix = '.yaml'
    for name in files_list:
        if name.lower().endswith(suffix.lower()):
            out_list.append(name[0:-len(suffix)])

    return out_list


def make_hi_lo(text: str, hi_suffix='hi', lo_suffix='lo'):
    hi = f"{text}_{hi_suffix}"
    lo = f"{text}_{lo_suffix}"
    return [hi, lo]


def ui():
    def autonomic_range_slider_row(
            shared_id: str,
            label='',
            minimum=float(0), maximum=float(1),
            step=0.05,
            lo_desc='min arousal', hi_desc='max arousal'
    ):
        [hi, lo] = make_hi_lo(shared_id)

        if not label:
            label = shared_id

        if params[lo] and type(params[lo]) is (float or int):
            value_lo = params[lo]
        else:
            value_lo = 0

        if params[hi] and type(params[hi]) is (float or int):
            value_hi = params[hi]
        else:
            value_hi = 0

        with gr.Row():
            shared.gradio[lo] = gr.Slider(
                label=f"{label} ({lo_desc})", minimum=minimum, maximum=maximum, step=step, value=value_lo, elem_id=lo)
            shared.gradio[hi] = gr.Slider(
                label=f"{label} ({hi_desc})", minimum=minimum, maximum=maximum, step=step, value=value_hi, elem_id=hi)

    with gr.Row():
        button_a = gr.Button(value='Autonomic Update', elem_id='load_autonomic')
        shared.gradio['print_debug'] = gr.Checkbox(
            value=params['print_debug'], label='Print debug information to console')
        buffer_switch = gr.Number(interactive=False, visible=False)

    with gr.Accordion(label='Parameter Ranges', open=False):
        autonomic_range_slider_row('temp', minimum=0.05, maximum=2)
        autonomic_range_slider_row('typical_p')
        autonomic_range_slider_row('repetition_penalty', maximum=2)
        autonomic_range_slider_row('encoder_repetition_penalty', maximum=2)
        autonomic_range_slider_row('penalty_alpha', maximum=5)
        autonomic_range_slider_row('top_k', maximum=75, step=1)

    with gr.Row():
        select_range = gr.Dropdown(label='Load a saved parameter range', choices=list_files('param_ranges'),
                                   value='Select range to load', interactive=True)
    with gr.Row():
        save_text = gr.Textbox(label='Parameter range name')
        save_btn = gr.Button(value='Save')

    def autonomic_change_method(shared_id: str):
        [hi, lo] = make_hi_lo(shared_id)
        shared.gradio[lo].change(lambda x: params.update({lo: x}), shared.gradio[lo], None)
        shared.gradio[hi].change(lambda x: params.update({hi: x}), shared.gradio[hi], None)

    def update_dropdown(v):
        return gr.Dropdown.update(choices=list_files('param_ranges'), value=v)

    shared.gradio['print_debug'].change(lambda x: params.update({"print_debug": x}), shared.gradio['print_debug'], None)
    button_a.click(autonomic_update, [shared.gradio['textbox'], buffer_switch]) \
        .then(which_params, buffer_switch, [shared.gradio['preset_menu'], buffer_switch])

    autonomic_change_method('temp')
    autonomic_change_method('typical_p')
    autonomic_change_method('repetition_penalty')
    autonomic_change_method('encoder_repetition_penalty')
    autonomic_change_method('penalty_alpha')
    autonomic_change_method('top_k')

    def autonomic_event_update():
        for name in list(params.keys()):
            (gr.update(value=params[name]), select_range, shared.gradio[name])

    select_range.select(lambda x: load_params(x), select_range, save_text).then(
        lambda x: gr.update(value=x), select_range, save_text).then(
        lambda x: autonomic_event_update(), select_range, save_text
    )

    save_btn.click(lambda x: save_params(x), save_text, select_range).then(
        update_dropdown, save_text, select_range).then(
        lambda x: autonomic_event_update(), save_text, select_range
    )
