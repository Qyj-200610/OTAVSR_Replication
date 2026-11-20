# Copyright (c) OTAVSR. All rights reserved.
"""LRS3 annotation parsing (the two CSV layouts seen in the wild)."""
import pytest

from otavsr.datasets import LRS3Dataset


def test_parse_row_three_fields():
    name, frames, text = LRS3Dataset._parse_row('lrs3/00001,40,hello, world')
    assert name == 'lrs3/00001'
    assert frames == 40
    assert text == 'hello, world'  # commas inside the transcript survive


def test_parse_row_five_fields():
    line = 'lrs3,lrs3/00002,25,-,second transcript'
    name, frames, text = LRS3Dataset._parse_row(line)
    assert name == 'lrs3/00002'
    assert frames == 25
    assert text == 'second transcript'


def test_parse_row_rejects_garbage():
    with pytest.raises(ValueError):
        LRS3Dataset._parse_row('this is not a csv row')


def test_load_data_list(tmp_path):
    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels / 'train.csv').write_text(
        'lrs3/00001,40,hello world\n'
        'lrs3/00002,25,second utterance\n',
        encoding='utf-8')

    dataset = LRS3Dataset(
        data_root=str(tmp_path),
        ann_file='labels/train.csv',
        pipeline=[],
        modality='audiovisual')

    assert len(dataset) == 2
    first = dataset.get_data_info(0)
    assert first['sample_id'] == 'lrs3/00001'
    assert first['num_frames'] == 40
    assert first['gt_text'] == 'hello world'
    assert first['video_path'].replace('\\', '/').endswith(
        'lrs3/lrs3_video_seg16s/lrs3/00001.mp4')
    assert first['audio_path'].replace('\\', '/').endswith('.wav')


def test_empty_annotation_raises(tmp_path):
    labels = tmp_path / 'labels'
    labels.mkdir()
    (labels / 'train.csv').write_text('\n\n', encoding='utf-8')
    with pytest.raises(RuntimeError):
        LRS3Dataset(
            data_root=str(tmp_path),
            ann_file='labels/train.csv',
            pipeline=[])
