//! REST bridge endpoints and OpenAPI 3.1.0 specification generator for OmniGet Server.

use axum::{
    body::Bytes,
    extract::Request,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::tools::{
    facebook_post::{extract_facebook_post, FacebookArgs, FacebookExtractError},
    instagram_post::{extract_instagram_post, InstagramArgs, InstagramExtractError},
    media_info::{extract_media_info, MediaInfoArgs},
    pdf_text::{extract_pdf_text, PdfTextArgs},
    web_markdown::web_to_markdown,
    x_extract::{extract_post, extract_thread, XArgs, XExtractError},
};

// ============================================================================
// Query & Payload Models
// ============================================================================

/// JSON payload for `POST /api/web/markdown`.
#[derive(Debug, Deserialize)]
pub struct WebMarkdownPayload {
    pub url: String,
}

// ============================================================================
// Helpers
// ============================================================================

/// Parses query string from request URI into a key-value HashMap.
fn parse_query(req: &Request) -> HashMap<String, String> {
    let query_str = req.uri().query().unwrap_or("");
    url::form_urlencoded::parse(query_str.as_bytes())
        .into_owned()
        .collect()
}

///// Parses JSON request body bytes into the specified type `T`.
///
/// Returns `StatusCode::BAD_REQUEST` (400) on empty body, malformed JSON, or missing required fields.
fn parse_json_payload<T: serde::de::DeserializeOwned>(bytes: Bytes) -> Result<T, Box<Response>> {
    if bytes.is_empty() {
        return Err(Box::new(
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "ok": false,
                    "error": "Missing JSON request body"
                })),
            )
                .into_response(),
        ));
    }

    serde_json::from_slice::<T>(&bytes).map_err(|err| {
        Box::new(
            (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "ok": false,
                    "error": format!("Malformed JSON payload or invalid schema: {}", err)
                })),
            )
                .into_response(),
        )
    })
}

// ============================================================================
// REST Handlers
// ============================================================================

// ── Web to Markdown ──────────────────────────────────────────────────────────

/// `GET /api/web/markdown?url=...`
pub async fn web_markdown_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let url = query.get("url").map(|s| s.trim()).unwrap_or("");
    if url.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Query parameter 'url' cannot be empty"
            })),
        )
            .into_response();
    }

    execute_web_markdown(url).await
}

/// `POST /api/web/markdown` (`{"url": "..."}`)
pub async fn web_markdown_post_handler(body: Bytes) -> Response {
    let payload: WebMarkdownPayload = match parse_json_payload(body) {
        Ok(p) => p,
        Err(resp) => return *resp,
    };

    let url = payload.url.trim();
    if url.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Field 'url' cannot be empty"
            })),
        )
            .into_response();
    }

    execute_web_markdown(url).await
}

async fn execute_web_markdown(url: &str) -> Response {
    match web_to_markdown(url).await {
        Ok(res) => (
            StatusCode::OK,
            Json(json!({
                "title": res.title,
                "markdown": res.markdown,
                "url": res.url
            })),
        )
            .into_response(),
        Err(err) => {
            let msg = err.to_string();
            let status = if msg.contains("URL") || msg.contains("scheme") || msg.contains("Invalid") {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, Json(json!({ "ok": false, "error": msg }))).into_response()
        }
    }
}

// ── PDF Text Extraction ──────────────────────────────────────────────────────

/// `GET /api/pdf/text?path=...&pages=...`
pub async fn pdf_text_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let args = PdfTextArgs {
        path: query.get("path").cloned(),
        url: query.get("url").cloned(),
        pages: query.get("pages").cloned(),
    };

    execute_pdf_text(args).await
}

/// `POST /api/pdf/text` (`{"path": "...", "pages": "..."}`)
pub async fn pdf_text_post_handler(body: Bytes) -> Response {
    let args: PdfTextArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    execute_pdf_text(args).await
}

async fn execute_pdf_text(mut args: PdfTextArgs) -> Response {
    let target = args
        .path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            args.url
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");

    if target.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Missing required parameter: 'path' or 'url'"
            })),
        )
            .into_response();
    }

    if args.path.as_deref().map(str::trim).filter(|s| !s.is_empty()).is_none() {
        args.path = None;
    }

    match extract_pdf_text(args).await {
        Ok(res) => (
            StatusCode::OK,
            Json(json!({
                "text": res.text,
                "pages": res.pages
            })),
        )
            .into_response(),
        Err(err) => {
            let msg = err.to_string();
            let status = if msg.contains("does not exist")
                || msg.contains("Missing required")
                || msg.contains("Invalid page")
                || msg.contains("Invalid start page")
                || msg.contains("Invalid end page")
                || msg.contains("No pages selected")
                || msg.contains("Page numbers")
                || msg.contains("out of bounds")
            {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, Json(json!({ "ok": false, "error": msg }))).into_response()
        }
    }
}

// ── Twitter/X Post ───────────────────────────────────────────────────────────

/// `GET /api/x/post?url=...` or `?id=...`
pub async fn x_post_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let target = query
        .get("url")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            query
                .get("id")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_x_post(target).await
}

/// `POST /api/x/post` (`{"url": "..."}`)
pub async fn x_post_post_handler(body: Bytes) -> Response {
    let args: XArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    let target = args
        .url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            args.id
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_x_post(target).await
}

async fn execute_x_post(target: &str) -> Response {
    if target.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Missing required parameter 'url' or 'id'"
            })),
        )
            .into_response();
    }

    match extract_post(target).await {
        Ok(post) => match serde_json::to_value(post) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(XExtractError::InvalidInput(msg)) | Err(XExtractError::InvalidDomain(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::NotFound(msg)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::UpstreamError(msg)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::Timeout(msg)) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
    }
}

// ── Twitter/X Thread ─────────────────────────────────────────────────────────

/// `GET /api/x/thread?url=...` or `?id=...`
pub async fn x_thread_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let target = query
        .get("url")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            query
                .get("id")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_x_thread(target).await
}

/// `POST /api/x/thread` (`{"url": "..."}`)
pub async fn x_thread_post_handler(body: Bytes) -> Response {
    let args: XArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    let target = args
        .url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            args.id
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_x_thread(target).await
}

async fn execute_x_thread(target: &str) -> Response {
    if target.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Missing required parameter 'url' or 'id'"
            })),
        )
            .into_response();
    }

    match extract_thread(target).await {
        Ok(thread) => match serde_json::to_value(thread) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(XExtractError::InvalidInput(msg)) | Err(XExtractError::InvalidDomain(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::NotFound(msg)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::UpstreamError(msg)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(XExtractError::Timeout(msg)) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
    }
}

// ── Media Info ───────────────────────────────────────────────────────────────

/// `GET /api/media/info?url=...`
pub async fn media_info_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let url = query.get("url").map(|s| s.trim()).unwrap_or("");
    execute_media_info(url).await
}

/// `POST /api/media/info` (`{"url": "..."}`)
pub async fn media_info_post_handler(body: Bytes) -> Response {
    let args: MediaInfoArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    execute_media_info(&args.url).await
}

async fn execute_media_info(url: &str) -> Response {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Media URL cannot be empty"
            })),
        )
            .into_response();
    }

    match extract_media_info(MediaInfoArgs {
        url: trimmed.to_string(),
    })
    .await
    {
        Ok(info) => match serde_json::to_value(info) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(err) => {
            let msg = err.to_string();
            let status = if msg.contains("empty")
                || msg.contains("Malformed")
                || msg.contains("Unsupported URL scheme")
            {
                StatusCode::BAD_REQUEST
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, Json(json!({ "ok": false, "error": msg }))).into_response()
        }
    }
}

// ── Instagram Post ──────────────────────────────────────────────────────────

/// `GET /api/instagram/post?url=...` or `?shortcode=...`
pub async fn instagram_post_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let target = query
        .get("url")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            query
                .get("shortcode")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_instagram_post(target).await
}

/// `POST /api/instagram/post` (`{"url": "..."}` or `{"shortcode": "..."}`)
pub async fn instagram_post_post_handler(body: Bytes) -> Response {
    let args: InstagramArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    let target = args
        .url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            args.shortcode
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_instagram_post(target).await
}

async fn execute_instagram_post(target: &str) -> Response {
    if target.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Missing required parameter 'url' or 'shortcode'"
            })),
        )
            .into_response();
    }

    match extract_instagram_post(target).await {
        Ok(post) => match serde_json::to_value(post) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(InstagramExtractError::InvalidInput(msg))
        | Err(InstagramExtractError::InvalidDomain(msg))
        | Err(InstagramExtractError::UnsupportedUrl(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(InstagramExtractError::NotFound(msg)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(InstagramExtractError::PrivateOrLoginRequired(msg)) => (
            StatusCode::FORBIDDEN,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(InstagramExtractError::UpstreamError(msg)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(InstagramExtractError::Timeout(msg)) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
    }
}

// ── Facebook Post ───────────────────────────────────────────────────────────

/// `GET /api/facebook/post?url=...` or `?id=...`
pub async fn facebook_post_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let target = query
        .get("url")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            query
                .get("id")
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_facebook_post(target).await
}

/// `POST /api/facebook/post` (`{"url": "..."}` or `{"id": "..."}`)
pub async fn facebook_post_post_handler(body: Bytes) -> Response {
    let args: FacebookArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    let target = args
        .url
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            args.id
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
        })
        .unwrap_or("");
    execute_facebook_post(target).await
}

async fn execute_facebook_post(target: &str) -> Response {
    if target.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "ok": false,
                "error": "Missing required parameter 'url' or 'id'"
            })),
        )
            .into_response();
    }

    match extract_facebook_post(target).await {
        Ok(post) => match serde_json::to_value(post) {
            Ok(v) => (StatusCode::OK, Json(v)).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "ok": false, "error": e.to_string() })),
            )
                .into_response(),
        },
        Err(FacebookExtractError::InvalidInput(msg))
        | Err(FacebookExtractError::InvalidDomain(msg))
        | Err(FacebookExtractError::UnsupportedUrl(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(FacebookExtractError::NotFound(msg)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(FacebookExtractError::PrivateOrLoginWall(msg)) => (
            StatusCode::FORBIDDEN,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(FacebookExtractError::UpstreamError(msg)) => (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
        Err(FacebookExtractError::Timeout(msg)) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json!({ "ok": false, "error": msg })),
        )
            .into_response(),
    }
}

// ── Universal Notion Clipper ────────────────────────────────────────────────

/// `GET /api/clip?url=...`
pub async fn clip_get_handler(req: Request) -> Response {
    let query = parse_query(&req);
    let url = query.get("url").cloned().unwrap_or_default();
    let token = query.get("token").cloned();
    let notion_database_id = query.get("notion_database_id").cloned();
    let notion_api_key = query.get("notion_api_key").cloned();
    let openrouter_api_key = query.get("openrouter_api_key").cloned();
    let model = query.get("model").cloned();
    let force = query.get("force").map(|v| v == "true" || v == "1");

    let args = crate::tools::clip::ClipArgs {
        url,
        token,
        notion_database_id,
        notion_api_key,
        openrouter_api_key,
        model,
        force,
    };

    let result = crate::tools::clip::execute_clip(args).await;
    let status = if result.ok {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    (status, Json(result)).into_response()
}

/// `POST /api/clip` (`{"url": "...", "notion_database_id": "...", ...}`)
pub async fn clip_post_handler(body: Bytes) -> Response {
    let args: crate::tools::clip::ClipArgs = match parse_json_payload(body) {
        Ok(a) => a,
        Err(resp) => return *resp,
    };

    let result = crate::tools::clip::execute_clip(args).await;
    let status = if result.ok {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };
    (status, Json(result)).into_response()
}


// ============================================================================
// OpenAPI 3.1.0 Specification Generator
// ============================================================================

/// `GET /openapi.json` handler.
pub async fn openapi_json_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(build_openapi_spec()))
}

/// Generates a valid OpenAPI 3.1.0 JSON document.
pub fn build_openapi_spec() -> Value {
    let mut spec = json!({
        "openapi": "3.1.0",
        "info": {
            "title": "OmniGet Standalone MCP Server",
            "version": "0.1.0",
            "description": "Standalone headless Model Context Protocol (MCP) and REST extraction server for X/Twitter posts & threads, universal web-to-markdown, PDF text extraction, and media metadata."
        },
        "servers": [
            {
                "url": "/",
                "description": "Current server instance"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check probe",
                    "description": "Public unauthenticated health check endpoint returning HTTP 200 OK.",
                    "responses": {
                        "200": {
                            "description": "Server is alive and operational",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "ok": { "type": "boolean" },
                                            "version": { "type": "string" }
                                        },
                                        "required": ["ok", "version"]
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/mcp": {
                "post": {
                    "summary": "Streamable HTTP MCP protocol endpoint",
                    "description": "Handles JSON-RPC 2.0 requests (initialize, ping, tools/list, tools/call) and notifications.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "description": "JSON-RPC 2.0 request payload",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "jsonrpc": { "type": "string", "example": "2.0" },
                                        "id": {
                                            "oneOf": [
                                                { "type": "string" },
                                                { "type": "integer" }
                                            ]
                                        },
                                        "method": { "type": "string" },
                                        "params": { "type": "object" }
                                    },
                                    "required": ["jsonrpc", "method"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "JSON-RPC 2.0 response result or error"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                }
            },
            "/sse": {
                "get": {
                    "summary": "MCP Server-Sent Events handshake",
                    "description": "Initiates a persistent Server-Sent Events (SSE) session stream.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "responses": {
                        "200": {
                            "description": "SSE stream established"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                }
            },
            "/messages": {
                "post": {
                    "summary": "MCP SSE message dispatcher",
                    "description": "Dispatches JSON-RPC 2.0 messages to an active SSE session identified by sessionId query parameter.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "sessionId",
                            "in": "query",
                            "required": true,
                            "schema": { "type": "string", "format": "uuid" },
                            "description": "Session UUID allocated during SSE handshake"
                        }
                    ],
                    "requestBody": {
                        "required": true,
                        "description": "JSON-RPC 2.0 payload",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "jsonrpc": { "type": "string", "example": "2.0" },
                                        "id": {
                                            "oneOf": [
                                                { "type": "string" },
                                                { "type": "integer" }
                                            ]
                                        },
                                        "method": { "type": "string" }
                                    },
                                    "required": ["jsonrpc", "method"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "202": {
                            "description": "Message accepted for dispatch to SSE stream"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                }
            },
            "/api/web/markdown": {
                "get": {
                    "summary": "Universal Web to Clean Markdown (GET)",
                    "description": "Fetches a public web page, strips boilerplate/ads/tracking parameters, and converts article text into clean Markdown.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": true,
                            "schema": { "type": "string" },
                            "description": "Public URL of the web page to scrape"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Clean Markdown extraction output",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "title": { "type": "string" },
                                            "markdown": { "type": "string" },
                                            "url": { "type": "string" }
                                        },
                                        "required": ["title", "markdown", "url"]
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request: missing or invalid URL"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream network error or HTTP non-2xx status"
                        }
                    }
                },
                "post": {
                    "summary": "Universal Web to Clean Markdown (POST)",
                    "description": "Fetches a public web page, strips boilerplate/ads/tracking parameters, and converts article text into clean Markdown.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string", "description": "Public URL to scrape" }
                                    },
                                    "required": ["url"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Clean Markdown extraction output",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "title": { "type": "string" },
                                            "markdown": { "type": "string" },
                                            "url": { "type": "string" }
                                        },
                                        "required": ["title", "markdown", "url"]
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request: missing or invalid URL parameter"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream network error"
                        }
                    }
                }
            },
            "/api/pdf/text": {
                "get": {
                    "summary": "Pure-Rust PDF text extraction (GET)",
                    "description": "Extracts readable plain text and page count from a local or remote PDF document.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "path",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Local filesystem path to PDF"
                        },
                        {
                            "name": "url",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Remote HTTP/HTTPS URL to PDF"
                        },
                        {
                            "name": "pages",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Optional page range filter (e.g. 1-3, all)"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "PDF text extracted successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": { "type": "string" },
                                            "pages": { "type": "integer" }
                                        },
                                        "required": ["text", "pages"]
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request: invalid file, missing parameter, or range error"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                },
                "post": {
                    "summary": "Pure-Rust PDF text extraction (POST)",
                    "description": "Extracts readable plain text and page count from a local or remote PDF document.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "path": { "type": "string", "description": "Local file path" },
                                        "url": { "type": "string", "description": "Remote PDF URL" },
                                        "url_or_path": { "type": "string", "description": "Local file path or remote PDF URL" },
                                        "pages": { "type": "string", "description": "Page range filter (e.g. 1-3)" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "PDF text extracted successfully",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "text": { "type": "string" },
                                            "pages": { "type": "integer" }
                                        },
                                        "required": ["text", "pages"]
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request: missing parameters or nonexistent file"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                }
            },
            "/api/x/post": {
                "get": {
                    "summary": "Twitter/X Post Extraction (GET)",
                    "description": "Extracts tweet content, author metadata, metrics, and media attachments via FxTwitter v2.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "X/Twitter post URL"
                        },
                        {
                            "name": "id",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Numeric status ID"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "XPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain or non-numeric status ID"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream FxTwitter API failure"
                        }
                    }
                },
                "post": {
                    "summary": "Twitter/X Post Extraction (POST)",
                    "description": "Extracts tweet content, author metadata, metrics, and media attachments via FxTwitter v2.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string" },
                                        "id": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "XPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain or non-numeric status ID"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream FxTwitter API failure"
                        }
                    }
                }
            },
            "/api/x/thread": {
                "get": {
                    "summary": "Twitter/X Thread Unrolling (GET)",
                    "description": "Unrolls full conversation thread by author returning focal post and sequential tweets.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" }
                        },
                        {
                            "name": "id",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Unrolled Thread object"
                        },
                        "400": {
                            "description": "Bad request: invalid domain or status ID"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream FxTwitter API failure"
                        }
                    }
                },
                "post": {
                    "summary": "Twitter/X Thread Unrolling (POST)",
                    "description": "Unrolls full conversation thread by author returning focal post and sequential tweets.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string" },
                                        "id": { "type": "string" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Unrolled Thread object"
                        },
                        "400": {
                            "description": "Bad request: invalid domain or status ID"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: upstream FxTwitter API failure"
                        }
                    }
                }
            },
            "/api/media/info": {
                "get": {
                    "summary": "Universal Media Metadata Extraction (GET)",
                    "description": "Queries metadata (title, author, platform, duration, available qualities) for 1,800+ supported sites or direct audio/video streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": true,
                            "schema": { "type": "string" },
                            "description": "Media item URL"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "MediaInfo object"
                        },
                        "400": {
                            "description": "Bad request: missing or malformed URL"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: media metadata extraction failure"
                        }
                    }
                },
                "post": {
                    "summary": "Universal Media Metadata Extraction (POST)",
                    "description": "Queries metadata (title, author, platform, duration, available qualities) for 1,800+ supported sites or direct audio/video streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string", "description": "Media URL" }
                                    },
                                    "required": ["url"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "MediaInfo object"
                        },
                        "400": {
                            "description": "Bad request: missing or malformed URL"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "502": {
                            "description": "Bad gateway: media metadata extraction failure"
                        }
                    }
                }
            },
            "/api/instagram/post": {
                "get": {
                    "summary": "Instagram Post Extraction (GET)",
                    "description": "Extracts public Instagram post, Reel, or carousel metadata, author, caption, hashtags, and media streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Public Instagram post URL (e.g. https://www.instagram.com/p/C_abc123/)"
                        },
                        {
                            "name": "shortcode",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Instagram shortcode"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "InstagramPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain, unsupported URL, or invalid shortcode"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "403": {
                            "description": "Forbidden: post is private or requires login"
                        },
                        "404": {
                            "description": "Not found: post does not exist or was deleted"
                        },
                        "502": {
                            "description": "Bad gateway: upstream extraction failure"
                        }
                    }
                },
                "post": {
                    "summary": "Instagram Post Extraction (POST)",
                    "description": "Extracts public Instagram post, Reel, or carousel metadata, author, caption, hashtags, and media streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string", "description": "Public Instagram post URL" },
                                        "shortcode": { "type": "string", "description": "Instagram shortcode" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "InstagramPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain, unsupported URL, or invalid shortcode"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "403": {
                            "description": "Forbidden: post is private or requires login"
                        },
                        "404": {
                            "description": "Not found: post does not exist or was deleted"
                        },
                        "502": {
                            "description": "Bad gateway: upstream extraction failure"
                        }
                    }
                }
            },
            "/api/facebook/post": {
                "get": {
                    "summary": "Facebook Post Extraction (GET)",
                    "description": "Extracts public Facebook post, Reel, or Watch video metadata, author, caption, images, and direct video streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Public Facebook post URL (e.g. https://www.facebook.com/page/posts/123456789)"
                        },
                        {
                            "name": "id",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Facebook post or video ID"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "FacebookPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain, unsupported URL, or empty input"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "403": {
                            "description": "Forbidden: post is private or blocked by login wall"
                        },
                        "404": {
                            "description": "Not found: post does not exist or was deleted"
                        },
                        "502": {
                            "description": "Bad gateway: upstream extraction failure"
                        }
                    }
                },
                "post": {
                    "summary": "Facebook Post Extraction (POST)",
                    "description": "Extracts public Facebook post, Reel, or Watch video metadata, author, caption, images, and direct video streams.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string", "description": "Public Facebook post URL" },
                                        "id": { "type": "string", "description": "Facebook post or video ID" }
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "FacebookPost object with author and media details"
                        },
                        "400": {
                            "description": "Bad request: invalid domain, unsupported URL, or empty input"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        },
                        "403": {
                            "description": "Forbidden: post is private or blocked by login wall"
                        },
                        "404": {
                            "description": "Not found: post does not exist or was deleted"
                        },
                        "502": {
                            "description": "Bad gateway: upstream extraction failure"
                        }
                    }
                }
            },
            "/api/clip": {
                "get": {
                    "summary": "Clip Web/Social Page to Notion (GET)",
                    "description": "Scrapes target URL, summarizes with AI (Groq/OpenRouter), and saves to Notion database with duplicate detection.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "parameters": [
                        {
                            "name": "url",
                            "in": "query",
                            "required": true,
                            "schema": { "type": "string" },
                            "description": "Public URL to scrape, summarize, and clip to Notion"
                        },
                        {
                            "name": "force",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "boolean" },
                            "description": "Force re-clipping even if URL was previously saved in Notion database"
                        },
                        {
                            "name": "notion_database_id",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Notion Database ID override"
                        },
                        {
                            "name": "notion_api_key",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "Notion Integration API Key override"
                        },
                        {
                            "name": "openrouter_api_key",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "LLM API Key override"
                        },
                        {
                            "name": "model",
                            "in": "query",
                            "required": false,
                            "schema": { "type": "string" },
                            "description": "LLM model name override"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "ClipResult object with title, notion_url, duplicate, and status"
                        },
                        "400": {
                            "description": "Bad request: invalid input, scraping failure, or Notion error"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                },
                "post": {
                    "summary": "Clip Web/Social Page to Notion (POST)",
                    "description": "Scrapes target URL, summarizes with AI (Groq/OpenRouter), and saves to Notion database with duplicate detection.",
                    "security": [
                        { "BearerAuth": [] }
                    ],
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": { "type": "string", "description": "Public URL to scrape and clip" },
                                        "force": { "type": "boolean", "description": "Force re-clipping even if duplicate" },
                                        "notion_database_id": { "type": "string", "description": "Notion Database ID override" },
                                        "notion_api_key": { "type": "string", "description": "Notion Integration API Key override" },
                                        "openrouter_api_key": { "type": "string", "description": "LLM API Key override" },
                                        "model": { "type": "string", "description": "LLM model name override" }
                                    },
                                    "required": ["url"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "ClipResult object with title, notion_url, duplicate, and status"
                        },
                        "400": {
                            "description": "Bad request: invalid input, scraping failure, or Notion error"
                        },
                        "401": {
                            "description": "Unauthorized: missing or invalid Bearer token"
                        }
                    }
                }
            }
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "Timing-safe Bearer token authorization header"
                }
            }
        }
    });

    if let Some(paths) = spec.get_mut("paths").and_then(Value::as_object_mut) {
        if let Some(p) = paths.get("/api/web/markdown").cloned() {
            paths.insert("/api/markdown".to_string(), p);
        }
        if let Some(p) = paths.get("/api/pdf/text").cloned() {
            paths.insert("/api/pdf".to_string(), p);
        }
        if let Some(p) = paths.get("/api/media/info").cloned() {
            paths.insert("/api/media".to_string(), p);
        }
        if let Some(p) = paths.get("/api/instagram/post").cloned() {
            paths.insert("/api/instagram".to_string(), p);
        }
        if let Some(p) = paths.get("/api/facebook/post").cloned() {
            paths.insert("/api/facebook".to_string(), p);
        }
    }

    spec
}

/// Public image proxy handler that streams images from Facebook lookaside/CDN to Notion.
pub async fn image_proxy_handler(req: Request) -> Response {
    use axum::http::header;

    let query_map = parse_query(&req);
    let raw_target_url = match query_map.get("url") {
        Some(u) if !u.trim().is_empty() => u.trim(),
        _ => {
            return (StatusCode::BAD_REQUEST, "Missing 'url' query parameter").into_response();
        }
    };

    // Clean any HTML entities from the target URL (&amp; -> &, \u0026 -> &)
    let target_url = raw_target_url
        .replace("&amp;", "&")
        .replace(r"\u0026", "&");

    // SSRF / security check: only proxy Facebook/Instagram CDN image URLs
    let is_allowed = target_url.contains("fbsbx.com")
        || target_url.contains("fbcdn.net")
        || target_url.contains("cdninstagram.com")
        || target_url.contains("instagram.com");

    if !is_allowed {
        return (StatusCode::FORBIDDEN, "Forbidden proxy target").into_response();
    }

    let is_instagram = target_url.contains("instagram.com") || target_url.contains("cdninstagram.com");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .unwrap_or_default();

    let browser_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
    let fb_crawler_ua = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)";

    // Attempt 1: For Instagram use Instagram Referer + Browser UA; for Facebook use Crawler UA
    let (ua_1, referer_1) = if is_instagram {
        (browser_ua, Some("https://www.instagram.com/"))
    } else {
        (fb_crawler_ua, None)
    };

    let mut req_builder = client
        .get(&target_url)
        .header(header::USER_AGENT, ua_1)
        .header(header::ACCEPT, "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8");
    if let Some(ref_val) = referer_1 {
        req_builder = req_builder.header(header::REFERER, ref_val);
    }
    let mut resp = req_builder.send().await;

    // Attempt 2: If attempt 1 failed or returned non-200, try browser UA with NO referer
    if resp.as_ref().map(|r| !r.status().is_success()).unwrap_or(true) {
        if let Ok(retry_resp) = client
            .get(&target_url)
            .header(header::USER_AGENT, browser_ua)
            .header(header::ACCEPT, "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8")
            .send()
            .await
        {
            if retry_resp.status().is_success() {
                resp = Ok(retry_resp);
            }
        }
    }

    // Attempt 3: If still non-200, try opposite UA strategy
    if resp.as_ref().map(|r| !r.status().is_success()).unwrap_or(true) {
        let (ua_3, referer_3) = if is_instagram {
            (fb_crawler_ua, None)
        } else {
            (browser_ua, Some("https://www.facebook.com/"))
        };
        let mut req_builder3 = client
            .get(&target_url)
            .header(header::USER_AGENT, ua_3)
            .header(header::ACCEPT, "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8");
        if let Some(ref_val) = referer_3 {
            req_builder3 = req_builder3.header(header::REFERER, ref_val);
        }
        if let Ok(retry_resp3) = req_builder3.send().await {
            if retry_resp3.status().is_success() {
                resp = Ok(retry_resp3);
            }
        }
    }

    let resp = match resp {
        Ok(r) => r,
        Err(e) => {
            return (StatusCode::BAD_GATEWAY, format!("Upstream fetch failed: {}", e)).into_response();
        }
    };

    let status = resp.status();
    if !status.is_success() {
        return (StatusCode::BAD_GATEWAY, format!("Upstream returned HTTP {}", status)).into_response();
    }

    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("image/jpeg")
        .to_string();

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to read image body: {}", e)).into_response();
        }
    };

    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, "public, max-age=2592000, immutable".to_string()),
        ],
        bytes,
    )
        .into_response()
}
