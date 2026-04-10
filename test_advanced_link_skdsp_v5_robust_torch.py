import inspect
import torch

import advanced_link_skdsp_v5_robust_torch as m


def test_bit_helpers_roundtrip():
    data = b"abc123"
    bits = m.bytes_to_bits_msb(data)
    assert m.bits_to_bytes_msb(bits) == data
    assert m.parse_header_bytes(m.build_header_bytes(42)) == 42


def test_fec_helpers():
    bits = [0, 1, 1, 0, 1, 0, 0, 1]
    rep = m.rep3_encode_bits(bits)
    soft = torch.tensor([1.0 if b else -1.0 for b in rep], dtype=torch.float64, device=m.DEFAULT_DEVICE)
    assert m.rep3_decode_soft(soft)[: len(bits)] == bits
    conv = m.conv_encode(bits)
    softc = torch.tensor([1.0 if b else -1.0 for b in conv], dtype=torch.float64, device=m.DEFAULT_DEVICE)
    assert m.conv_decode_soft(softc)[: len(bits)] == bits


def test_signal_ops_and_grad():
    x = torch.randn(256, dtype=torch.float32, device=m.DEFAULT_DEVICE)
    x = torch.complex(x, torch.zeros_like(x)).requires_grad_(True)
    y = m.apply_carrier_frequency(x, 1e3, 1e6)
    z = m.robust_agc_and_blanking(y)
    loss = torch.mean(torch.abs(z) ** 2)
    loss.backward()
    assert x.grad is not None


def test_tx_rx_end_to_end_message():
    result = m.build_tx_iq_object(
        message="torch-e2e",
        fec=m.FEC_NONE,
        interleave=True,
        snr_db=35.0,
        noise_color="white",
        freq_offset=0.0,
        timing_offset=1.0,
        fading_mode="none",
        burst_probability=0.0,
    )
    rx = m.rx_command_iq(result.iq, result.metadata)
    assert rx["payload_len"] > 0
    assert rx["message"] == "torch-e2e"


def test_all_functions_smoke():
    # Smoke-test presence/signatures for all top-level callables declared in module.
    public = {
        name: obj
        for name, obj in vars(m).items()
        if inspect.isfunction(obj) and obj.__module__ == m.__name__ and not name.startswith("_")
    }
    # Ensure the expected API is present.
    expected = {
        'save_iq','load_iq','default_metadata_path','save_iq_metadata','load_iq_metadata','bytes_to_bits_msb','bits_to_bytes_msb',
        'prbs_bits','rrc_taps','measure_power','bpsk_map','upsample_and_shape','tx_waveform','apply_carrier_frequency',
        'apply_frequency_offset','apply_timing_offset_resample','resample_iq','apply_fading','add_impulsive_bursts','impair_iq',
        'robust_agc_and_blanking','lfsr_sequence','scramble_bits','rep3_encode_bits','rep3_decode_soft','block_interleave_bits',
        'block_deinterleave_soft','conv_encode','conv_decode_soft','build_header_bytes','parse_header_bytes',
        'build_payload_bytes_from_message','build_payload_bytes_from_random_bits','parse_payload_bytes','insert_pilots',
        'remove_pilots_soft','pilot_positions','build_tx_bitstream','build_tx_iq_object','save_tx_iq_object','coarse_frequency_acquire',
        'estimate_residual_cfo_from_preamble','apply_symbol_rate_cfo','matched_filter','extract_symbols_from_start',
        'design_symbol_equalizer_ls','apply_symbol_equalizer','apply_pilot_phase_tracking','choose_valid_header_from_copies',
        'try_decode_from_symbols','rx_command_iq','main'
    }
    assert expected.issubset(set(public.keys()))
