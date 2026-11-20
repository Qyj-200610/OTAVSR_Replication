# Copyright (c) OTAVSR. All rights reserved.
"""LRS3-TED dataset (paper §III).

The annotation files come from auto-avsr and are re-used by both official
baselines.  The CSV field order depends on which file you downloaded, so the
two layouts seen in the wild are both accepted:

* 3 fields: ``<rel_path>,<num_frames>,<transcript>``
* 5 fields: ``<dataset>,<rel_path>,<num_frames>,<unused>,<transcript>``

Print a couple of parsed rows before a long run (``tools/analysis_tools/
browse_dataset.py``): a wrong field mapping silently trains the model on
garbage transcripts.
"""
import os.path as osp
from typing import List, Tuple

from mmengine.dataset import BaseDataset
from mmengine.fileio import list_from_file
from mmengine.logging import MMLogger

from otavsr.registry import DATASETS

__all__ = ['LRS3Dataset']


@DATASETS.register_module()
class LRS3Dataset(BaseDataset):
    """LRS3-TED with mouth-ROI videos produced by auto-avsr.

    Args:
        data_root (str): root that contains ``labels/`` and ``lrs3/``.
        ann_file (str): CSV with one utterance per line.
        video_subdir (str): subdirectory of the 96x96 mouth crops.
        audio_subdir (str): subdirectory of the 16 kHz wavs.
        pipeline (list): data pipeline.
        modality (str): ``'audiovisual'`` / ``'audio'``; stored per sample.
        test_mode (bool): disable training-time augmentation.
        serialize_data (bool): serialise the data list (mmpretrain default).
            Note that with this on, ``dataset.data_list`` is emptied -- use
            ``dataset.get_data_info(i)`` (the frame sampler does).
    """

    #: Paper: arXiv:2607.09001 -- deliberately empty: mmengine treats any
    #: string value in ``METAINFO`` as a path to a meta-info file and warns
    #: "not a meta file, simply parsed as meta information" for each of them.
    METAINFO = dict()

    def __init__(self,
                 data_root: str,
                 ann_file: str = 'labels/lrs3_train_transcript_lengths_seg16s_LLM_lowercase_25.csv',
                 video_subdir: str = 'lrs3/lrs3_video_seg16s',
                 audio_subdir: str = 'lrs3/lrs3_audio_seg16s',
                 pipeline: List[dict] = (),
                 modality: str = 'audiovisual',
                 test_mode: bool = False,
                 serialize_data: bool = True,
                 lazy_init: bool = False,
                 max_refetch: int = 1000) -> None:
        self.modality = modality
        self.video_subdir = video_subdir
        self.audio_subdir = audio_subdir
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            metainfo=self.METAINFO,
            test_mode=test_mode,
            serialize_data=serialize_data,
            lazy_init=lazy_init,
            max_refetch=max_refetch)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_row(line: str) -> Tuple[str, int, str]:
        """Accept both the 3-field and the 5-field annotation layouts."""
        parts = line.split(',')
        if len(parts) >= 3 and parts[1].strip().isdigit():
            return parts[0].strip(), int(parts[1]), ','.join(parts[2:]).strip()
        if len(parts) >= 5 and parts[2].strip().isdigit():
            return parts[1].strip(), int(parts[2]), ','.join(parts[4:]).strip()
        raise ValueError(
            f'cannot parse annotation row {line!r}: expected 3 or 5 '
            'comma-separated fields (path, num_frames, transcript).')

    def load_data_list(self) -> List[dict]:
        ann_path = osp.join(self.data_root, self.ann_file)
        data_list = []
        for line in list_from_file(ann_path):
            line = line.strip()
            if not line:
                continue
            name, num_frames, text = self._parse_row(line)
            video = osp.join(self.data_root, self.video_subdir, name)
            if not video.endswith(('.mp4', '.avi', '.mkv')):
                video += '.mp4'
            audio = osp.join(self.data_root, self.audio_subdir, name)
            if not audio.endswith('.wav'):
                audio += '.wav'
            data_list.append(
                dict(
                    sample_id=name,
                    video_path=video,
                    audio_path=audio,
                    num_frames=num_frames,
                    gt_text=text,
                    label=text,
                    modality=self.modality))

        if len(data_list) == 0:
            raise RuntimeError(f'no sample found in {ann_path!r}')
        MMLogger.get_current_instance().info(
            f'Loaded {len(data_list)} samples from {ann_path}')
        return data_list
