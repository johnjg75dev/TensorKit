//! HTML report generation from analysis data.
//!
//! Renders `Analysis` to an interactive HTML report using the built-in
//! template with Chart.js for interactive charts.

use crate::analysis::{render_svg, Analysis, BlockRole};
use crate::error::Result;
use std::path::Path;

/// Built-in HTML template (embedded at compile time).
const DEFAULT_TEMPLATE: &str = include_str!("../templates/report.html");

/// Render an `Analysis` to HTML using the provided template.
///
/// If `template_path` is `None`, uses the built-in template.
/// Supports both interactive (Chart.js) and static (SVG) chart rendering.
pub fn render_html_report(
    analysis: &Analysis,
    template_path: Option<&Path>,
) -> Result<String> {
    let template = if let Some(path) = template_path {
        std::fs::read_to_string(path).map_err(crate::error::Error::Io)?
    } else {
        DEFAULT_TEMPLATE.to_string()
    };

    let mut html = template;

    // -- Basic metadata --
    let model_name = analysis.model_name.as_deref().unwrap_or("Unknown");
    let arch = analysis.architecture.as_deref().unwrap_or("Unknown");
    let total_mb = analysis.total_bytes as f64 / 1_048_576.0;
    let total_size = if analysis.total_bytes > 0 {
        format!("{:.2} MB", total_mb)
    } else {
        "N/A".into()
    };
    let est_after_mb = analysis.estimated_bytes_after_prune as f64 / 1_048_576.0;

    html = html.replace("{{TITLE}}", "TensorKit Analysis Report");
    html = html.replace("{{DATE}}", &chrono_lite_date());
    html = html.replace("{{MODEL_NAME}}", model_name);
    html = html.replace("{{ARCHITECTURE}}", arch);
    html = html.replace("{{BLOCK_COUNT}}", &analysis.blocks.len().to_string());
    html = html.replace("{{TOTAL_SIZE}}", &total_size);

    // -- Summary body --
    let mut summary = String::new();
    summary.push_str(&format!("tensors:       {}\n", analysis.total_tensors));
    summary.push_str(&format!("total bytes:   {:.2} MB\n", total_mb));
    summary.push_str(&format!("sample/tensor: {}\n", analysis.sample_per_tensor));
    summary.push_str(&format!("blocks:        {}\n", analysis.blocks.len()));
    summary.push_str(&format!(
        "recommend:     {} blocks -> {:.2} MB after prune\n",
        analysis.recommendation_count, est_after_mb
    ));
    for (name, role) in &ROLES {
        let n = analysis.blocks.iter().filter(|b| b.role == *role).count();
        if n > 0 {
            summary.push_str(&format!("  {:<8} {}\n", name, n));
        }
    }
    html = html.replace("{{SUMMARY_BODY}}", &escape_html(&summary));

    // -- Recommendation --
    html = html.replace(
        "{{RECOMMENDATION_COUNT}}",
        &analysis.recommendation_count.to_string(),
    );
    html = html.replace(
        "{{RECOMMENDATION}}",
        &format!("{:?}", analysis.recommendation),
    );
    html = html.replace(
        "{{ESTIMATED_SIZE}}",
        &format!("{:.2} MB", est_after_mb),
    );

    // -- Block rows --
    let block_rows: String = analysis
        .blocks
        .iter()
        .map(|b| {
            let status = if b.removable > 0.5 { "Prunable" } else { "Keep" };
            let tag = if b.removable > 0.5 { "tag-prunable" } else { "tag-keep" };
            format!(
                r#"<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.3}</td><td><span class="tag {}">{}</span></td></tr>"#,
                escape_html(&b.label),
                b.role.as_str(),
                b.tensor_count,
                b.total_bytes as f64 / 1_048_576.0,
                b.removable,
                tag,
                status,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    html = html.replace("{{BLOCK_ROWS}}", &block_rows);

    // -- Tensor rows --
    let tensor_rows: String = analysis
        .blocks
        .iter()
        .flat_map(|b| &b.tensors)
        .map(|t| {
            format!(
                r#"<tr><td><code>{}</code></td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.4}</td><td>{:.2}</td></tr>"#,
                escape_html(&t.name),
                t.stats.mean,
                t.stats.std,
                t.stats.sparsity_abs,
                t.stats.outlier_ratio,
                t.stats.entropy_bits,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let tensor_rows = if tensor_rows.is_empty() {
        "<tr><td colspan=\"6\">(no tensor data)</td></tr>".into()
    } else {
        tensor_rows
    };
    html = html.replace("{{TENSOR_ROWS}}", &tensor_rows);
    html = html.replace("{{SAMPLE_PER_TENSOR}}", &analysis.sample_per_tensor.to_string());

    // -- Charts section (SVG fallback) --
    let charts = crate::analysis::analysis_to_charts(analysis);
    let charts_svg: String = if charts.is_empty() {
        String::new()
    } else {
        let svgs: Vec<String> = charts
            .iter()
            .map(|chart| {
                format!(
                    r#"<div class="chart-container"><h3>{}</h3>{}</div>"#,
                    escape_html(&chart.title),
                    render_svg(chart),
                )
            })
            .collect();
        format!(
            r#"<section class="card"><h2>Charts</h2>{}</section>"#,
            svgs.join("\n"),
        )
    };
    html = html.replace("{{CHARTS_SECTION}}", &charts_svg);

    // -- Interactive chart data (JSON for Chart.js) --
    let chart_json = build_chart_json(analysis);
    html = html.replace("{{CHART_JSON}}", &chart_json);

    // -- Spectra canvases --
    let mut spectra_canvases = String::new();
    let mut spec_idx = 0usize;
    for b in &analysis.blocks {
        for spec in b.spectra.values() {
            if spec.is_empty() {
                continue;
            }
            spectra_canvases.push_str(&format!(
                r#"<div class="chart-container"><canvas id="chart-spectrum-{idx}"></canvas></div>"#,
                idx = spec_idx,
            ));
            spec_idx += 1;
        }
    }
    if spectra_canvases.is_empty() {
        spectra_canvases = String::from("<p class=\"muted\">No spectra data available.</p>");
    }
    html = html.replace("{{SPECTRA_CANVAS}}", &spectra_canvases);

    // -- Dry-run section (placeholder, filled by CLI context) --
    html = html.replace("{{DRY_RUN_SECTION}}", "");

    Ok(html)
}

/// Build the JSON data structure for Chart.js interactive charts.
fn build_chart_json(analysis: &Analysis) -> String {
    use std::collections::BTreeMap;

    // Amax values sorted descending
    let amax: Vec<f64> = {
        let mut v: Vec<f64> = analysis
            .blocks
            .iter()
            .flat_map(|b| b.tensors.iter().map(|t| t.stats.abs_max))
            .filter(|x| x.is_finite() && *x > 0.0)
            .collect();
        v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        v
    };

    // Role counts
    let role_counts: Vec<BTreeMap<&str, serde_json::Value>> = ROLES
        .iter()
        .filter_map(|(name, role)| {
            let n = analysis.blocks.iter().filter(|b| b.role == *role).count();
            if n > 0 {
                let mut map = BTreeMap::new();
                map.insert("label", serde_json::Value::String(name.to_string()));
                map.insert("count", serde_json::Value::Number(n.into()));
                Some(map)
            } else {
                None
            }
        })
        .collect();

    // Spectra
    let spectra: Vec<BTreeMap<&str, serde_json::Value>> = analysis
        .blocks
        .iter()
        .flat_map(|b| &b.spectra)
        .filter(|(_, spec)| !spec.is_empty())
        .map(|(name, spec)| {
            let mut map = BTreeMap::new();
            map.insert("label", serde_json::Value::String(name.clone()));
            map.insert(
                "values",
                serde_json::Value::Array(
                    spec.iter().map(|v| serde_json::Value::Number(
                        serde_json::Number::from_f64(*v as f64).unwrap_or(serde_json::Number::from(0))
                    )).collect()
                ),
            );
            map
        })
        .collect();

    let data = serde_json::json!({
        "amax": amax,
        "roleCounts": role_counts,
        "spectra": spectra,
    });

    data.to_string()
}

const ROLES: [(&str, BlockRole); 5] = [
    ("embed", BlockRole::Embedding),
    ("output", BlockRole::OutputHead),
    ("norm", BlockRole::FinalNorm),
    ("block", BlockRole::Block),
    ("other", BlockRole::Other),
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

fn chrono_lite_date() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Howard Hinnant's civil date algorithm (public domain)
    let days = secs / 86400;
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../tests/unit/report.rs"]
mod tests;
