import json
import math
from os import listdir, makedirs
from os.path import isfile, join, isdir

import gradio as gr
import modules.shared as shared
import numpy as np
from transformers import pipeline

temp_range = [0.4, 1.2]
typical_p_range = [0.8, 0.2]
repetition_penalty_range = [1.15, 1.1]
encoder_repetition_penalty_range = [1.05, 1]

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                      top_k=None, device="cpu")

# Default parameters which can be customized in settings.json of webui
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


def sigmoid(z):
    a = 1 / (1 + (np.exp((-z))))
    return a


def autonomic_map(dict_list):
    out = []
    mtx = [0, 0, 0, 0, 0, 0]
    print_d('--------------')
    print_d('Raw Sentiment Output')
    print_d('--------------')
    for i, d in enumerate(dict_list):
        print_d(f"{d['label']}: {d['score']}")
    print_d('--------------')
    print_d('Sentiment Component Contributions')
    print_d('--------------')
    # Anger
    # anger = sigmoid(15 * mtx[0] - 10) + 0.2 * np.sin(math.pi*mtx[0])**2

    # Anger
    mtx[0] = [d['score'] for d in dict_list if d['label'] == 'anger'][0]
    anger = mtx[0] ** 3
    out.append(anger)
    print_d(f'Anger: {np.round(anger, 3)}')

    # Disgust
    mtx[1] = [d['score'] for d in dict_list if d['label'] == 'disgust'][0]
    disgust = mtx[1]
    out.append(disgust)
    print_d(f'Disgust: {np.round(disgust, 3)}')

    # Fear
    mtx[2] = [d['score'] for d in dict_list if d['label'] == 'fear'][0]
    fear = mtx[2] ** 2
    out.append(fear)
    print_d(f'Fear: {np.round(fear, 3)}')

    # Joy
    mtx[3] = [d['score'] for d in dict_list if d['label'] == 'joy'][0]
    joy = 0.75 * (mtx[3] + 0.1) * np.cos(math.pi * mtx[3]) ** 2
    out.append(joy)
    print_d(f'Joy: {np.round(joy, 3)}')

    # Neutral is the 4th value

    # Sadness
    mtx[4] = [d['score'] for d in dict_list if d['label'] == 'sadness'][0]
    sadness = 0.75 * (mtx[4] + 0.1) * np.cos(math.pi * mtx[4]) ** 2
    out.append(sadness)
    print_d(f'Sadness: {np.round(sadness, 3)}')

    # Surprise
    mtx[5] = [d['score'] for d in dict_list if d['label'] == 'surprise'][0]
    surprise = mtx[5]
    out.append(surprise)
    print_d(f'Surprise: {np.round(surprise, 3)}')

    # out = anger + disgust + fear + joy + sadness + surprise

    # Normalize if desired
    final = np.sum(out)
    # final = np.sum(out) / np.sum(mtx)

    # Ceiling
    if final > 1:
        final = 1

    print_d('--------------')
    print_d(f'Autonomic coefficient: {np.round(final, 2)}')
    print_d('--------------')

    return final


def range_bias(val, bias):
    out = val[0] + bias * (val[1] - val[0])
    return out


def parameter_map(bias, toggle):
    params_new = [range_bias([params['temp_lo'], params['temp_hi']], bias),
                  range_bias([params['typical_p_lo'], params['typical_p_hi']], bias),
                  range_bias([params['repetition_penalty_lo'], params['repetition_penalty_hi']], bias),
                  range_bias([params['encoder_repetition_penalty_lo'], params['encoder_repetition_penalty_hi']], bias),
                  range_bias([params['penalty_alpha_lo'], params['penalty_alpha_hi']], bias),
                  range_bias([params['top_k_lo'], params['top_k_hi']], bias),
                  ]

    if toggle == 0:
        fname = "presets/Autonomic_Buffer_A.txt"
    else:
        fname = "presets/Autonomic_Buffer_B.txt"

    with open(fname, "w") as text_file:
        print(f'temperature={params_new[0]}\n', file=text_file)
        print(f'typical_p={params_new[1]}\n', file=text_file)
        print(f'repetition_penalty={params_new[2]}\n', file=text_file)
        print(f'encoder_repetition_penalty={params_new[3]}\n', file=text_file)
        print(f'penalty_alpha={params_new[4]}\n', file=text_file)
        print(f'top_k={params_new[5]}\n', file=text_file)


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

    parameter_map(autonomic_map(emotions), buffer)


def save_params(filename):
    with open(f'param_ranges/{filename}.json', 'w') as f:
        json.dump(params, f)
        print_d('--------------')
        print_d(f'Parameter ranges saved to \'param_ranges/{filename}.json\':')
        print_d('--------------')
        for k, v in params.items():
            print_d(f"{k}: {v:.2f}")


def load_params(filename):
    with open(f'param_ranges/{filename}.json') as f:
        params.update(json.load(f))
        print_d('--------------')
        print_d(f'Parameter ranges after loading \'param_ranges/{filename}.json\':')
        print_d('--------------')
        for k, v in params.items():
            print_d(f"{k}: {v:.2f}")


def list_files(path):
    if not isdir(path):
        makedirs(path)
    files_list = [f for f in listdir(path) if isfile(join(path, f))]
    out_list = []
    suffix = '.json'
    for name in files_list:
        if name.lower().endswith(suffix.lower()):
            out_list.append(name[0:-len(suffix)])

    return out_list


'''
def activate(x):
    params.update({"active": x})
    if x:
        return [x, 'autonomic_params']
    else:
        return [x, 'LLaMA-Precise']



def preset_modifier(text):
    if params['active']:
        return 'autonomic_params'
    else:
        return text
'''


def ui():
    with gr.Row():
        button_a = gr.Button(value='Autonomic Update', elem_id='load_autonomic')
        # active = gr.Checkbox(value=params['active'], label='Enable autonomic parameter management')
        shared.gradio['print_debug'] = gr.Checkbox(value=params['print_debug'], label='Print debug information to console')
        buffer_switch = gr.Number(interactive=False, visible=False)
    with gr.Accordion(label='Parameter Ranges', open=False):
        with gr.Row():
            shared.gradio['temp_lo'] = gr.Slider(label='temperature (min arousal)', minimum=0.05, maximum=2, step=0.05,
                                                 value=params['temp_lo'], elem_id='temp_lo')
            shared.gradio['temp_hi'] = gr.Slider(label='temperature (max arousal)', minimum=0.05, maximum=2, step=0.05,
                                                 value=params['temp_hi'], elem_id='temp_hi')
        with gr.Row():
            shared.gradio['typical_p_lo'] = gr.Slider(label='typical_p (min arousal)', minimum=0, maximum=1, step=0.05,
                                                      value=params['typical_p_lo'], elem_id='typical_p_lo')
            shared.gradio['typical_p_hi'] = gr.Slider(label='typical_p (max arousal)', minimum=0, maximum=1, step=0.05,
                                                      value=params['typical_p_hi'], elem_id='typical_p_hi')
        with gr.Row():
            shared.gradio['repetition_penalty_lo'] = gr.Slider(label='repetition_penalty (min arousal)', minimum=0,
                                                               maximum=2, step=0.05,
                                                               value=params['repetition_penalty_lo'],
                                                               elem_id='repetition_penalty_lo')
            shared.gradio['repetition_penalty_hi'] = gr.Slider(label='repetition_penalty (max arousal)', minimum=0,
                                                               maximum=2, step=0.05,
                                                               value=params['repetition_penalty_hi'],
                                                               elem_id='repetition_penalty_hi')
        with gr.Row():
            shared.gradio['encoder_repetition_penalty_lo'] = gr.Slider(label='encoder_repetition_penalty (min arousal)',
                                                                       minimum=0,
                                                                       maximum=2, step=0.05,
                                                                       value=params['encoder_repetition_penalty_lo'],
                                                                       elem_id='encoder_repetition_penalty_lo')
            shared.gradio['encoder_repetition_penalty_hi'] = gr.Slider(label='encoder_repetition_penalty (max arousal)',
                                                                       minimum=0,
                                                                       maximum=2, step=0.05,
                                                                       value=params['encoder_repetition_penalty_hi'],
                                                                       elem_id='encoder_repetition_penalty_hi')
        with gr.Row():
            shared.gradio['penalty_alpha_lo'] = gr.Slider(label='penalty_alpha (min arousal)', minimum=0, maximum=5,
                                                          step=0.05,
                                                          value=params['penalty_alpha_lo'],
                                                          elem_id='penalty_alpha_lo')
            shared.gradio['penalty_alpha_hi'] = gr.Slider(label='penalty_alpha (max arousal)', minimum=0, maximum=5,
                                                          step=0.05,
                                                          value=params['penalty_alpha_hi'],
                                                          elem_id='penalty_alpha_hi')
        with gr.Row():
            shared.gradio['top_k_lo'] = gr.Slider(label='top_k (min arousal)', minimum=0, maximum=75, step=0.05,
                                                  value=params['top_k_lo'], elem_id='top_k_lo')
            shared.gradio['top_k_hi'] = gr.Slider(label='top_k (max arousal)', minimum=0, maximum=75, step=0.05,
                                                  value=params['top_k_hi'], elem_id='top_k_hi')
    with gr.Row():
        select_range = gr.Dropdown(label='Load a saved parameter range', choices=list_files('param_ranges'),
                                   value='Select range to load', interactive=True)
    with gr.Row():
        save_txt = gr.Textbox(label='Parameter range name')
        save_btn = gr.Button(value='Save')

    def update_elements(element):
        return params[element.elem_id]

    def update_dropdown(v):
        return gr.Dropdown.update(choices=list_files('param_ranges'), value=v)

    # active.change(lambda x: activate(x), active, shared.gradio['preset_menu'])
    shared.gradio['print_debug'].change(lambda x: params.update({"print_debug": x}), shared.gradio['print_debug'], None)
    button_a.click(autonomic_update, [shared.gradio['textbox'], buffer_switch]) \
        .then(which_params, buffer_switch, [shared.gradio['preset_menu'], buffer_switch])

    shared.gradio['temp_lo'].change(lambda x: params.update({"temp_lo": x}), shared.gradio['temp_lo'], None)
    shared.gradio['temp_hi'].change(lambda x: params.update({"temp_hi": x}), shared.gradio['temp_hi'], None)
    shared.gradio['typical_p_lo'].change(lambda x: params.update({"typical_p_lo": x}), shared.gradio['typical_p_lo'], None)
    shared.gradio['typical_p_hi'].change(lambda x: params.update({"typical_p_hi": x}), shared.gradio['typical_p_hi'], None)
    shared.gradio['repetition_penalty_lo'].change(lambda x: params.update({"repetition_penalty_lo": x}),
                                                  shared.gradio['repetition_penalty_lo'], None)
    shared.gradio['repetition_penalty_hi'].change(lambda x: params.update({"repetition_penalty_hi": x}),
                                                  shared.gradio['repetition_penalty_hi'], None)
    shared.gradio['encoder_repetition_penalty_lo'].change(lambda x: params.update({"encoder_repetition_penalty_lo": x}),
                                                          shared.gradio['encoder_repetition_penalty_lo'], None)
    shared.gradio['encoder_repetition_penalty_hi'].change(lambda x: params.update({"encoder_repetition_penalty_hi": x}),
                                                          shared.gradio['encoder_repetition_penalty_hi'], None)
    shared.gradio['penalty_alpha_lo'].change(lambda x: params.update({"penalty_alpha_lo": x}), shared.gradio['penalty_alpha_lo'], None)
    shared.gradio['penalty_alpha_hi'].change(lambda x: params.update({"penalty_alpha_hi": x}), shared.gradio['penalty_alpha_hi'], None)
    shared.gradio['top_k_lo'].change(lambda x: params.update({"top_k_lo": x}), shared.gradio['top_k_lo'], None)
    shared.gradio['top_k_hi'].change(lambda x: params.update({"top_k_hi": x}), shared.gradio['top_k_hi'], None)

    select_range.select(lambda x: load_params(x), select_range, save_txt) \
        .then(lambda x: gr.update(value=x), select_range, save_txt) \
        .then(lambda x: gr.update(value=params['temp_lo']), None, shared.gradio['temp_lo']) \
        .then(lambda x: gr.update(value=params['temp_hi']), None, shared.gradio['temp_hi']) \
        .then(lambda x: gr.update(value=params['typical_p_lo']), None, shared.gradio['typical_p_lo']) \
        .then(lambda x: gr.update(value=params['typical_p_hi']), None, shared.gradio['typical_p_hi']) \
        .then(lambda x: gr.update(value=params['repetition_penalty_lo']), None, shared.gradio['repetition_penalty_lo']) \
        .then(lambda x: gr.update(value=params['repetition_penalty_hi']), None, shared.gradio['repetition_penalty_hi']) \
        .then(lambda x: gr.update(value=params['encoder_repetition_penalty_lo']), None, shared.gradio['encoder_repetition_penalty_lo']) \
        .then(lambda x: gr.update(value=params['encoder_repetition_penalty_hi']), None, shared.gradio['encoder_repetition_penalty_hi']) \
        .then(lambda x: gr.update(value=params['penalty_alpha_lo']), None, shared.gradio['penalty_alpha_lo']) \
        .then(lambda x: gr.update(value=params['penalty_alpha_hi']), None, shared.gradio['penalty_alpha_hi']) \
        .then(lambda x: gr.update(value=params['top_k_lo']), None, shared.gradio['top_k_lo']) \
        .then(lambda x: gr.update(value=params['top_k_hi']), None, shared.gradio['top_k_hi'])
    save_btn.click(lambda x: save_params(x), save_txt, None) \
        .then(update_dropdown, save_txt, select_range) \
        .then(lambda x: gr.update(value=params['temp_lo']), None, shared.gradio['temp_lo']) \
        .then(lambda x: gr.update(value=params['temp_hi']), None, shared.gradio['temp_hi']) \
        .then(lambda x: gr.update(value=params['typical_p_lo']), None, shared.gradio['typical_p_lo']) \
        .then(lambda x: gr.update(value=params['typical_p_hi']), None, shared.gradio['typical_p_hi']) \
        .then(lambda x: gr.update(value=params['repetition_penalty_lo']), None, shared.gradio['repetition_penalty_lo']) \
        .then(lambda x: gr.update(value=params['repetition_penalty_hi']), None, shared.gradio['repetition_penalty_hi']) \
        .then(lambda x: gr.update(value=params['encoder_repetition_penalty_lo']), None, shared.gradio['encoder_repetition_penalty_lo']) \
        .then(lambda x: gr.update(value=params['encoder_repetition_penalty_hi']), None, shared.gradio['encoder_repetition_penalty_hi']) \
        .then(lambda x: gr.update(value=params['penalty_alpha_lo']), None, shared.gradio['penalty_alpha_lo']) \
        .then(lambda x: gr.update(value=params['penalty_alpha_hi']), None, shared.gradio['penalty_alpha_hi']) \
        .then(lambda x: gr.update(value=params['top_k_lo']), None, shared.gradio['top_k_lo']) \
        .then(lambda x: gr.update(value=params['top_k_hi']), None, shared.gradio['top_k_hi'])
