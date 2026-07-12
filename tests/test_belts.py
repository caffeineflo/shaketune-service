import matplotlib.pyplot as plt
import numpy as np
import pytest

from shaketune.graph_creators.base_models import GraphMetadata
from shaketune.graph_creators.computations.belts_computation import BeltsComputation
from shaketune.graph_creators.computation_results import BeltsResult, SignalData
from shaketune.graph_creators.plotters.belts_plotter import BeltsPlotter
from tests.conftest import make_csv_bytes, make_gzip_csv


def test_offset_table_compares_each_belt_signal():
    paired_peaks = [((1, 10.0, 100.0), (2, 40.0, 60.0))]
    signal1 = SignalData(
        freqs=np.array([0.0, 10.0, 20.0]),
        psd=np.array([0.0, 100.0, 80.0]),
        peaks=np.array([1]),
        paired_peaks=paired_peaks,
        unpaired_peaks=[],
    )
    signal2 = SignalData(
        freqs=np.array([0.0, 30.0, 40.0]),
        psd=np.array([0.0, 20.0, 60.0]),
        peaks=np.array([2]),
        paired_peaks=paired_peaks,
        unpaired_peaks=[],
    )
    result = BeltsResult(
        metadata=GraphMetadata(title='Test belts'),
        measurements=[{'name': 'belt-a', 'samples': []}, {'name': 'belt-b', 'samples': []}],
        signal1=signal1,
        signal2=signal2,
        signal1_belt='a',
        signal2_belt='b',
        kinematics=None,
        test_params=('PULSE-ONLY', None, None, None, None, None, None),
        max_freq=50.0,
        max_scale=None,
    )

    figure = BeltsPlotter().plot(result)
    try:
        table = figure.axes[0].tables[0]
        displayed_offsets = [table.get_celld()[(1, column)].get_text().get_text() for column in (1, 2)]
        assert displayed_offsets == ['30.0 Hz', '40.0 %']
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ('measurement_names', 'expected_labels'),
    [
        (
            ('raw_data_axis=1.000,-1.000_a', 'raw_data_axis=1.000,1.000_b'),
            ('A (axis 1,-1)', 'B (axis 1, 1)'),
        ),
        (
            ('belt_A_20260712_144333', 'belt_B_20260712_144333'),
            ('A (axis 1,-1)', 'B (axis 1, 1)'),
        ),
    ],
)
def test_computation_labels_belt_measurements(measurement_names, expected_labels, monkeypatch):
    measurements = [{'name': name, 'samples': [[0.0]]} for name in measurement_names]
    signal = SignalData(
        freqs=np.array([0.0, 1.0]),
        psd=np.array([0.0, 1.0]),
        peaks=np.array([], dtype=int),
        paired_peaks=[],
        unpaired_peaks=[],
    )
    computation = BeltsComputation(
        measurements=measurements,
        kinematics=None,
        max_freq=200.0,
        test_params=None,
        max_scale=None,
        st_version='test',
    )
    monkeypatch.setattr(computation, '_compute_signal_data', lambda *_: signal)

    result = computation.compute()

    assert (result.signal1_belt, result.signal2_belt) == expected_labels


class TestBeltsEndpoint:
    """POST /belts endpoint tests."""

    def test_standard_upload(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert data['type'] == 'belts'
        assert data['printer'] == 'default'
        assert data['url'].endswith('_belts.png')

    def test_busybox_separate_fields(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('file_a', ('raw_data_a.csv', csv, 'text/csv')),
            ('file_b', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 200
        assert resp.json()['type'] == 'belts'

    @pytest.mark.parametrize(
        ('form_data', 'expected_kinematics'),
        [({}, 'corexy'), ({'kinematics': 'corexz'}, 'corexz')],
    )
    def test_kinematics_passed_to_cli(self, client, mock_cli, form_data, expected_kinematics):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('file_a', ('raw_data_a.csv', csv, 'text/csv')),
            ('file_b', ('raw_data_b.csv', csv, 'text/csv')),
        ], data=form_data)

        cmd = mock_cli[0][0]
        kinematics_index = cmd.index('--kinematics')
        assert (resp.status_code, cmd[kinematics_index + 1]) == (200, expected_kinematics)

    def test_one_file_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 400
        assert 'requires 2 files' in resp.json()['detail']

    def test_no_files_returns_400(self, client, mock_cli):
        resp = client.post('/belts')
        assert resp.status_code == 400

    def test_custom_printer_and_timestamp(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ], data={'printer': 'k1v4', 'timestamp': '20260325_140000'})
        data = resp.json()
        assert data['printer'] == 'k1v4'
        assert '20260325_140000_belts.png' in data['url']
        assert '/results/k1v4/' in data['url']

    def test_gzip_upload(self, client, mock_cli):
        gz = make_gzip_csv()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv.gz', gz, 'application/gzip')),
            ('files', ('raw_data_b.csv.gz', gz, 'application/gzip')),
        ])
        assert resp.status_code == 200

    def test_cli_failure_returns_500(self, client, mock_cli_failure):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 500
        assert 'Analysis failed' in resp.json()['detail']

    def test_three_files_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_b.csv', csv, 'text/csv')),
            ('files', ('raw_data_c.csv', csv, 'text/csv')),
        ])
        assert resp.status_code == 400

    def test_duplicate_output_filename_returns_400(self, client, mock_cli):
        csv = make_csv_bytes()
        resp = client.post('/belts', files=[
            ('files', ('raw_data_a.csv', csv, 'text/csv')),
            ('files', ('raw_data_a.csv.gz', make_gzip_csv(), 'application/gzip')),
        ])
        assert resp.status_code == 400
