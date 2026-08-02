#[tokio::main]
async fn main() {
    let _observability =
        api_server::init_observability().expect("failed to initialize observability");
    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".to_string());
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|error| panic!("failed to bind {bind_addr}: {error}"));
    let router = api_server::build_router_from_env().expect("failed to configure provider");
    axum::serve(listener, router)
        .await
        .expect("api server failed");
}
