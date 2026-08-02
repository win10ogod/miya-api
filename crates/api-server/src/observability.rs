use axum::{extract::Request, middleware::Next, response::Response};
use opentelemetry::{global, propagation::Extractor, trace::TracerProvider as _};
use opentelemetry_sdk::{propagation::TraceContextPropagator, trace::SdkTracerProvider};
use tracing::{Instrument, info_span};
use tracing_opentelemetry::OpenTelemetrySpanExt;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct ObservabilityGuard {
    tracer_provider: Option<SdkTracerProvider>,
}

impl Drop for ObservabilityGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take() {
            let _ = provider.shutdown();
        }
    }
}

pub fn init() -> Result<ObservabilityGuard, String> {
    global::set_text_map_propagator(TraceContextPropagator::new());
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let fmt = tracing_subscriber::fmt::layer().with_target(false);
    let otlp_enabled = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok()
        || std::env::var("MIYA_OTEL_ENABLED")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
    if !otlp_enabled {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .try_init()
            .map_err(|error| error.to_string())?;
        return Ok(ObservabilityGuard {
            tracer_provider: None,
        });
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()
        .map_err(|error| format!("failed to build OTLP span exporter: {error}"))?;
    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();
    global::set_tracer_provider(provider.clone());
    let tracer = provider.tracer("miya-api");
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    tracing_subscriber::registry()
        .with(filter)
        .with(fmt)
        .with(telemetry)
        .try_init()
        .map_err(|error| error.to_string())?;
    Ok(ObservabilityGuard {
        tracer_provider: Some(provider),
    })
}

pub async fn trace_request(request: Request, next: Next) -> Response {
    let method = request.method().to_string();
    let route = request.uri().path().to_string();
    let parent_context = global::get_text_map_propagator(|propagator| {
        propagator.extract(&HeaderExtractor(request.headers()))
    });
    let span = info_span!(
        "http.request",
        otel.name = %format!("{method} {route}"),
        otel.kind = "server",
        http.request.method = %method,
        url.path = %route,
        http.response.status_code = tracing::field::Empty,
    );
    let _ = span.set_parent(parent_context);
    let response = next.run(request).instrument(span.clone()).await;
    span.record("http.response.status_code", response.status().as_u16());
    response
}

struct HeaderExtractor<'a>(&'a axum::http::HeaderMap);

impl Extractor for HeaderExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|value| value.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(axum::http::HeaderName::as_str).collect()
    }
}
