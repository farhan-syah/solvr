//! CPU signal processing tests.

use super::*;

fn setup() -> (CpuClient, <CpuRuntime as numr::runtime::Runtime>::Device) {
    let device = numr::runtime::cpu::CpuDevice::new();
    let client = CpuClient::new(device.clone());
    (client, device)
}

#[test]
fn test_convolve_full() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, -1.0], &[3], &device);

    let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

    assert_eq!(result.shape(), &[7]);
    let data: Vec<f64> = result.to_vec();
    // Expected: [1, 2, 2, 2, 2, -4, -5]
    assert!((data[0] - 1.0).abs() < 1e-6);
    assert!((data[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_convolve_same() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

    let result = client.convolve(&signal, &kernel, ConvMode::Same).unwrap();

    assert_eq!(result.shape(), &[5]);
}

#[test]
fn test_convolve_valid() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

    let result = client.convolve(&signal, &kernel, ConvMode::Valid).unwrap();

    assert_eq!(result.shape(), &[3]);
    let data: Vec<f64> = result.to_vec();
    // Sum of windows: [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert!((data[0] - 6.0).abs() < 1e-6);
    assert!((data[1] - 9.0).abs() < 1e-6);
    assert!((data[2] - 12.0).abs() < 1e-6);
}

#[test]
fn test_convolve2d() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &device,
    );
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], &device);

    let result = client.convolve2d(&signal, &kernel, ConvMode::Full).unwrap();

    assert_eq!(result.shape(), &[4, 4]);
}

#[test]
fn test_correlate() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device);

    let result = client.correlate(&signal, &kernel, ConvMode::Full).unwrap();

    assert_eq!(result.shape(), &[6]);
}

#[test]
fn test_stft_basic() {
    let (client, device) = setup();

    // Create a simple signal
    let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[256], &device);

    let result = client
        .stft(&signal_tensor, 64, Some(16), None, true, false)
        .unwrap();

    // Output should be complex with shape [n_frames, freq_bins]
    let freq_bins = 64 / 2 + 1; // 33
    let n_frames = (256 + 64 - 64) / 16 + 1; // 17

    assert_eq!(result.shape(), &[n_frames, freq_bins]);
}

#[test]
fn test_istft_reconstruction() {
    let (client, device) = setup();

    // Create a signal
    let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
    let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[256], &device);

    // STFT
    let stft_result = client
        .stft(&signal_tensor, 64, Some(16), None, true, false)
        .unwrap();

    // ISTFT
    let reconstructed = client
        .istft(&stft_result, Some(16), None, true, Some(256), false)
        .unwrap();

    assert_eq!(reconstructed.shape(), &[256]);

    // Check reconstruction is close to original
    let recon_data: Vec<f64> = reconstructed.to_vec();
    let mut max_error = 0.0f64;
    for (i, (&orig, &recon)) in signal.iter().zip(recon_data.iter()).enumerate() {
        let err = (orig - recon).abs();
        if err > max_error {
            max_error = err;
        }
        // Skip edges which have boundary effects
        if i > 32 && i < 224 {
            assert!(
                err < 0.1,
                "Reconstruction error at {}: {} vs {}",
                i,
                orig,
                recon
            );
        }
    }
}

#[test]
fn test_spectrogram() {
    let (client, device) = setup();

    let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
    let signal_tensor = Tensor::<CpuRuntime>::from_slice(&signal, &[512], &device);

    let result = client
        .spectrogram(&signal_tensor, 64, Some(32), None, 2.0)
        .unwrap();

    // Power spectrogram should be real-valued
    assert_eq!(result.dtype(), numr::dtype::DType::F64);

    // Check dimensions
    let freq_bins = 64 / 2 + 1;
    let n_frames = (512 + 64 - 64) / 32 + 1;
    assert_eq!(result.shape(), &[n_frames, freq_bins]);

    // All values should be non-negative (power)
    let data: Vec<f64> = result.to_vec();
    for val in data {
        assert!(val >= 0.0);
    }
}

#[test]
fn test_convolve_f32() {
    let (client, device) = setup();

    let signal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
    let kernel = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

    let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

    assert_eq!(result.dtype(), numr::dtype::DType::F32);
    assert_eq!(result.shape(), &[6]);
}
