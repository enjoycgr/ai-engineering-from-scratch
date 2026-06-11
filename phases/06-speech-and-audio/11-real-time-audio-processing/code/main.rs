// 课程：实时音频处理（phase 06 / 第 11 课）
// 主题：以 20 ms 帧流式传输 16 kHz 单声道正弦波，应用增益级
// 和 9 抽头低通 FIR 滤波器，测量每帧和聚合吞吐量。
// 这是每个语音助手在 VAD/ASR/TTS 下运行的内循环。
// 参考：
//   https://doc.rust-lang.org/std/time/struct.Instant.html
//   https://en.wikipedia.org/wiki/Finite_impulse_response
//   https://webrtc.googlesource.com/src/+/refs/heads/main/modules/audio_processing  (20 ms 帧约定)
// 构建：rustc --edition 2021 -O code/main.rs -o /tmp/lesson_audio && /tmp/lesson_audio

use std::f32::consts::PI;
use std::time::Instant;

const SAMPLE_RATE: u32 = 16_000;
const FRAME_MS: u32 = 20;
const FRAME_LEN: usize = (SAMPLE_RATE / 1000 * FRAME_MS) as usize; // 320 个样本
const TONE_HZ: f32 = 440.0;
const TOTAL_SECONDS: f32 = 2.0;
const GAIN_DB: f32 = -3.0;

// 9 抽头对称低通 FIR。手工调参，sum ~= 1.0 以保留 DC。
const FIR_TAPS: [f32; 9] = [
    0.02, 0.06, 0.12, 0.18, 0.24, 0.18, 0.12, 0.06, 0.02,
];

fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

fn synth_sine_frame(start_sample: u64, freq_hz: f32, sr: u32) -> Vec<f32> {
    let mut frame = Vec::with_capacity(FRAME_LEN);
    let two_pi_f_over_sr = 2.0 * PI * freq_hz / sr as f32;
    for n in 0..FRAME_LEN {
        let t = (start_sample + n as u64) as f32;
        frame.push((two_pi_f_over_sr * t).sin());
    }
    frame
}

fn apply_gain(frame: &mut [f32], gain_lin: f32) {
    for s in frame.iter_mut() {
        *s *= gain_lin;
    }
}

// 流式 FIR。`state` 跨帧边界携带最后 (taps-1) 个样本
// 使滤波器看到连续信号，而非带有边缘伪影的 20 ms 孤岛。
fn fir_streaming(frame: &mut [f32], taps: &[f32], state: &mut Vec<f32>) {
    let order = taps.len();
    let mut buf = Vec::with_capacity(state.len() + frame.len());
    buf.extend_from_slice(state);
    buf.extend_from_slice(frame);

    for n in 0..frame.len() {
        let mut acc = 0.0;
        for k in 0..order {
            acc += taps[k] * buf[n + order - 1 - k];
        }
        frame[n] = acc;
    }

    let keep = order - 1;
    state.clear();
    state.extend_from_slice(&buf[buf.len() - keep..]);
}

fn rms(frame: &[f32]) -> f32 {
    let sum_sq: f32 = frame.iter().map(|x| x * x).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

fn rms_dbfs(frame: &[f32]) -> f32 {
    let r = rms(frame).max(1e-10);
    20.0 * r.log10()
}

fn percentile(sorted_us: &[f64], pct: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_us.len() as f64 - 1.0) * pct).round() as usize;
    sorted_us[idx]
}

fn main() {
    let total_samples = (SAMPLE_RATE as f32 * TOTAL_SECONDS) as u64;
    let total_frames = (total_samples as usize) / FRAME_LEN;
    let gain_lin = db_to_linear(GAIN_DB);

    println!();
    println!("=== 实时音频基准测试 (Rust, 单线程) ===");
    println!();
    println!("采样率  : {} Hz", SAMPLE_RATE);
    println!("帧大小   : {} ms ({} 个样本)", FRAME_MS, FRAME_LEN);
    println!("流长度: {:.1} s ({} 帧)", TOTAL_SECONDS, total_frames);
    println!("音调         : {} Hz 正弦", TONE_HZ);
    println!("增益级   : {:+.1} dB", GAIN_DB);
    println!("FIR          : {} 抽头对称低通", FIR_TAPS.len());
    println!();

    let mut fir_state = vec![0.0f32; FIR_TAPS.len() - 1];
    let mut per_frame_us: Vec<f64> = Vec::with_capacity(total_frames);
    let mut rms_in_db = 0.0f32;
    let mut rms_out_db = 0.0f32;

    let wall = Instant::now();
    for f in 0..total_frames {
        let start_sample = (f * FRAME_LEN) as u64;
        let mut frame = synth_sine_frame(start_sample, TONE_HZ, SAMPLE_RATE);

        let t_frame = Instant::now();
        if f == 0 { rms_in_db = rms_dbfs(&frame); }

        apply_gain(&mut frame, gain_lin);
        fir_streaming(&mut frame, &FIR_TAPS, &mut fir_state);

        if f == 0 { rms_out_db = rms_dbfs(&frame); }
        per_frame_us.push(t_frame.elapsed().as_secs_f64() * 1e6);
    }
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;

    let mut sorted = per_frame_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);
    let p99 = percentile(&sorted, 0.99);
    let mean = per_frame_us.iter().sum::<f64>() / per_frame_us.len() as f64;

    let budget_us = (FRAME_MS as f64) * 1000.0;
    let headroom = budget_us / p99.max(1e-9);

    println!("每帧延迟 (us):");
    println!("  p50   {:>9.2}", p50);
    println!("  p95   {:>9.2}", p95);
    println!("  p99   {:>9.2}", p99);
    println!("  mean  {:>9.2}", mean);
    println!();
    println!("聚合:");
    println!("  墙上时间         {:>8.2} ms", wall_ms);
    println!("  实时预算   {:>8.2} ms ({} 帧 * {} ms)", total_frames as f64 * FRAME_MS as f64, total_frames, FRAME_MS);
    println!("  实时系数   {:>8.1}x   (wall/budget; 越低越快)", wall_ms / (total_frames as f64 * FRAME_MS as f64));
    println!("  p99 余量  {:>8.1}x   (budget / p99)", headroom);
    println!();
    println!("信号电平 (第 0 帧):");
    println!("  RMS 输入   {:>7.2} dBFS", rms_in_db);
    println!("  RMS 输出  {:>7.2} dBFS  (经过 {:+.1} dB 增益 + FIR)", rms_out_db, GAIN_DB);
    println!();

    if headroom >= 50.0 {
        println!("结论: 巨大余量。VAD + STT + LLM + TTS 都能塞进 20 ms 时隙。");
    } else if headroom >= 5.0 {
        println!("结论: 舒适余量。流式管线将能容纳。");
    } else {
        println!("结论: 太慢。在此 DSP 开销下管线将丢帧。");
    }
    println!();
}
