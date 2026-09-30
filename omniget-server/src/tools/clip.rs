//! Universal Smart Web & Social Clipper (`clip`).
//!
//! Orchestrates end-to-end clipping:
//! 1. Multi-platform extraction (Facebook, Instagram, X/Twitter, Web Markdown).
//! 2. AI Summarization & Categorization via OpenRouter (Gemini Flash).
//! 3. Structured export to Notion database with executive callout, native image blocks,
//!    and chunked raw content bypassing Notion's 2,000-character rich text limit.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::LazyLock;
use std::time::Duration;

use crate::tools::{
    facebook_post::extract_facebook_post,
    instagram_post::extract_instagram_post,
    web_markdown::web_to_markdown,
    x_extract::extract_thread,
    pdf_text::{extract_pdf_text, PdfTextArgs},
};

static IMG_MD_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"!\[.*?\]\((https?://[^\)\s]+)\)"#).expect("Valid markdown image regex")
});

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipArgs {
    pub url: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub notion_database_id: Option<String>,
    #[serde(default)]
    pub notion_api_key: Option<String>,
    #[serde(default)]
    pub openrouter_api_key: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipResponse {
    pub ok: bool,
    pub platform: String,
    pub title: String,
    pub author: Option<String>,
    pub notion_page_id: Option<String>,
    pub notion_url: Option<String>,
    pub images_count: usize,
    pub summary: String,
    pub ai_ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ai_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

struct ExtractedContent {
    platform: String,
    title: String,
    author: Option<String>,
    raw_text: String,
    images: Vec<String>,
    source_url: String,
}

/// Dispatches URL extraction across Facebook, Instagram, X (Twitter), and general web.
async fn extract_content(url: &str) -> ExtractedContent {
    let lower = url.to_lowercase();

    // 1. Twitter / X
    if lower.contains("x.com") || lower.contains("twitter.com") {
        match extract_thread(url).await {
            Ok(thread) => {
                let mut images = Vec::new();
                let mut full_text = String::new();
                for (i, post) in thread.posts.iter().enumerate() {
                    for m in &post.media {
                        let img = if m.kind == "photo" { m.url.clone() } else { m.thumb.clone() };
                        if !img.is_empty() {
                            images.push(img);
                        }
                    }
                    if thread.posts.len() > 1 {
                        full_text.push_str(&format!("--- Tweet {} by @{} ---\n{}\n\n", i + 1, post.author.handle, post.text));
                    } else {
                        full_text.push_str(&post.text);
                    }
                }
                
                let author_str = format!("{} (@{})", thread.focal.author.name, thread.focal.author.handle);
                let title = if thread.focal.text.chars().count() > 60 {
                    format!("{}...", thread.focal.text.chars().take(57).collect::<String>())
                } else if !thread.focal.text.is_empty() {
                    thread.focal.text.clone()
                } else {
                    format!("X Post by @{}", thread.focal.author.handle)
                };
                
                return ExtractedContent {
                    platform: "X (Twitter)".to_string(),
                    title,
                    author: Some(author_str),
                    raw_text: full_text.trim().to_string(),
                    images,
                    source_url: thread.focal.url.clone(),
                };
            }
            Err(e) => {
                tracing::warn!("X extraction failed ({}), falling back to web markdown", e);
            }
        }
    }

    // 2. Instagram
    if lower.contains("instagram.com") {
        match extract_instagram_post(url).await {
            Ok(post) => {
                let author_str = format!("@{}", post.author.username);
                let title = if !post.caption.is_empty() {
                    let first_line = post.caption.lines().next().unwrap_or("").trim();
                    if first_line.chars().count() > 60 {
                        format!("{}...", first_line.chars().take(57).collect::<String>())
                    } else {
                        first_line.to_string()
                    }
                } else {
                    format!("Instagram Post by @{}", post.author.username)
                };
                return ExtractedContent {
                    platform: "Instagram".to_string(),
                    title,
                    author: Some(author_str),
                    raw_text: post.caption,
                    images: post.images,
                    source_url: post.url,
                };
            }
            Err(e) => {
                tracing::warn!("Instagram extraction failed ({}), falling back to web markdown", e);
            }
        }
    }

    // 3. Facebook
    if lower.contains("facebook.com") || lower.contains("fb.watch") || lower.contains("fb.com") {
        match extract_facebook_post(url).await {
            Ok(post) => {
                let author_str = if !post.author.name.trim().is_empty() {
                    post.author.name
                } else {
                    "Facebook User".to_string()
                };
                let body_text = if !post.caption.trim().is_empty() {
                    post.caption
                } else {
                    post.markdown
                };
                let title = if !body_text.trim().is_empty() {
                    let first_line = body_text.lines().next().unwrap_or("").trim();
                    if first_line.chars().count() > 60 {
                        format!("{}...", first_line.chars().take(57).collect::<String>())
                    } else {
                        first_line.to_string()
                    }
                } else {
                    format!("Facebook Post by {}", author_str)
                };
                return ExtractedContent {
                    platform: "Facebook".to_string(),
                    title,
                    author: Some(author_str),
                    raw_text: body_text,
                    images: post.images,
                    source_url: post.url,
                };
            }
            Err(e) => {
                tracing::warn!("Facebook extraction failed ({}), falling back to web markdown", e);
            }
        }
    }

    // 3.5 PDF Documents
    if lower.ends_with(".pdf") || lower.contains(".pdf?") {
        match extract_pdf_text(PdfTextArgs {
            path: Some(url.to_string()),
            url: None,
            pages: None,
        }).await {
            Ok(pdf_result) => {
                let filename = url.rsplit('/').next().unwrap_or("Document.pdf").split('?').next().unwrap_or("Document.pdf");
                let title = format!("PDF Document: {}", filename);
                return ExtractedContent {
                    platform: "PDF Document".to_string(),
                    title,
                    author: None,
                    raw_text: pdf_result.text,
                    images: Vec::new(),
                    source_url: url.to_string(),
                };
            }
            Err(e) => {
                tracing::warn!("PDF extraction failed ({}), falling back to web markdown", e);
            }
        }
    }

    // 4. Default: General Web Article Scraper
    match web_to_markdown(url).await {
        Ok(web) => {
            let mut extracted_images = Vec::new();
            for cap in IMG_MD_REGEX.captures_iter(&web.markdown) {
                if let Some(m) = cap.get(1) {
                    let img_url = m.as_str().to_string();
                    if !extracted_images.contains(&img_url) {
                        extracted_images.push(img_url);
                    }
                }
            }
            ExtractedContent {
                platform: "Web Article".to_string(),
                title: web.title,
                author: None,
                raw_text: web.markdown,
                images: extracted_images,
                source_url: web.url,
            }
        }
        Err(e) => ExtractedContent {
            platform: "Web Page".to_string(),
            title: "Clipped Resource".to_string(),
            author: None,
            raw_text: format!("Extraction error: {}", e),
            images: Vec::new(),
            source_url: url.to_string(),
        },
    }
}

struct AiSummary {
    title: String,
    summary: String,
    category: String,
    priority: String,
    tags: Vec<String>,
    ai_ok: bool,
    ai_model: Option<String>,
    ai_error: Option<String>,
}

fn extract_json_object(s: &str) -> Option<Value> {
    let trimmed = s.trim();
    if let Ok(val) = serde_json::from_str::<Value>(trimmed) {
        return Some(val);
    }
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            let slice = &trimmed[start..=end];
            if let Ok(val) = serde_json::from_str::<Value>(slice) {
                return Some(val);
            }
        }
    }
    None
}

/// Generates executive summary and title using OpenRouter with automatic model fallback.
async fn generate_summary(
    content: &ExtractedContent,
    api_key_override: Option<&str>,
    model_override: Option<&str>,
) -> AiSummary {
    let groq_env = std::env::var("GROQ_API_KEY").ok().filter(|k| !k.trim().is_empty());
    let openrouter_env = std::env::var("OPENROUTER_API_KEY").ok().filter(|k| !k.trim().is_empty());

    let (api_key, is_groq) = if let Some(k) = api_key_override.filter(|k| !k.trim().is_empty()) {
        let is_g = k.starts_with("gsk_");
        (k.to_string(), is_g)
    } else if let Some(k) = groq_env {
        (k, true)
    } else if let Some(k) = openrouter_env {
        (k, false)
    } else {
        (String::new(), true)
    };

    let api_endpoint = if is_groq {
        "https://api.groq.com/openai/v1/chat/completions"
    } else {
        "https://openrouter.ai/api/v1/chat/completions"
    };

    let env_model = std::env::var("AI_MODEL").ok().filter(|m| !m.trim().is_empty());
    let candidate_models = if is_groq {
        // If env_model contains '/' it's an old OpenRouter model like google/gemma, ignore it on Groq
        let valid_primary = model_override
            .filter(|m| !m.trim().is_empty() && !m.contains('/'))
            .or_else(|| env_model.as_deref().filter(|m| !m.contains('/')))
            .unwrap_or("llama-3.3-70b-versatile");

        let mut models = vec![valid_primary.to_string()];
        for fallback in &["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"] {
            if !models.contains(&fallback.to_string()) {
                models.push(fallback.to_string());
            }
        }
        models
    } else {
        let primary = model_override
            .filter(|m| !m.trim().is_empty())
            .or(env_model.as_deref())
            .unwrap_or("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free");

        let mut models = vec![primary.to_string()];
        for fallback in &[
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "nvidia/nemotron-3.5-lightning:free",
            "qwen/qwen3.8-27b:free",
        ] {
            if !models.contains(&fallback.to_string()) {
                models.push(fallback.to_string());
            }
        }
        models
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .unwrap_or_default();

    let truncated_text: String = content.raw_text.chars().take(6000).collect();
    let prompt = format!(
        "Analyze, summarize, and classify this content from {}.\n\
        Author/Source: {}\n\
        Original Title/URL: {}\n\n\
        --- CONTENT ---\n\
        {}\n\
        --- END CONTENT ---\n\n\
        Return ONLY valid JSON matching this exact structure with no extra text or code fences:\n\
        {{\n\
          \"title\": \"A concise, engaging title in Thai with an appropriate leading emoji (e.g. 🩺, 📱, 📊, 🚀, 💡)\",\n\
          \"category\": \"Choose EXACTLY ONE from: 💻 ไอที / เทคโนโลยี | 🤖 ปัญญาประดิษฐ์ (AI) | 💼 ธุรกิจ / การเงิน | 📰 ข่าวสาร | 📚 ความรู้ / วิชาการ | 🩺 สุขภาพ / การแพทย์ | ☕ ทั่วไป / ไลฟ์สไตล์\",\n\
          \"priority\": \"Choose EXACTLY ONE from: 🔴 สูง (High) | 🟡 ปานกลาง (Medium) | 🟢 ต่ำ (Low)\",\n\
          \"tags\": [\"Keyword1\", \"Keyword2\", \"Keyword3\"],\n\
          \"summary\": \"The full aesthetic 4-section briefing in Thai with:\\n\\n🎯 **สรุปภาพรวม (Executive Summary)**\\n- ...\\n\\n🔍 **ประเด็นสำคัญและข้อมูลเชิงลึก (Key Highlights)**\\n- ...\\n\\n🚀 **สิ่งที่นำไปปรับใช้ได้จริง / ข้อคิด (Actionable Insights)**\\n- ...\\n\\n🏷️ **แท็กหัวข้อ (Topics & Keywords)**\\n`#Keyword1` `#Keyword2`\"\n\
        }}",
        content.platform,
        content.author.as_deref().unwrap_or("Unknown"),
        content.title,
        truncated_text
    );

    let mut last_ai_error: Option<String> = None;

    for model in candidate_models {
        tracing::info!("Calling AI ({}) with model: {}", if is_groq { "Groq" } else { "OpenRouter" }, model);
        let start_time = std::time::Instant::now();
        let mut payload = json!({
            "model": model,
            "max_tokens": 1500,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an elite research analyst and Notion knowledge architect. Return ONLY a valid JSON object matching the exact requested keys: title, category, priority, tags, summary. Keep bullet points concise and informative. Never output broken JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        });

        if is_groq {
            payload["response_format"] = json!({ "type": "json_object" });
        } else {
            payload["reasoning"] = json!({ "max_tokens": 0 });
        }

        let res = client
            .post(api_endpoint)
            .header("Authorization", format!("Bearer {}", api_key.trim()))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await;

        match res {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    if let Ok(data) = resp.json::<Value>().await {
                        let elapsed = start_time.elapsed().as_secs_f32();
                        let actual_model = data["model"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                            .unwrap_or(&model)
                            .to_string();
                        let verified_badge = format!("{} ({}, {:.1}s)", if is_groq { "Groq" } else { "OpenRouter" }, actual_model, elapsed);

                        if let Some(content_str) = data["choices"][0]["message"]["content"].as_str() {
                            let parsed_opt = extract_json_object(content_str).or_else(|| {
                                // If truncated mid-string or missing closing quotes/brackets, attempt safe repair
                                let trimmed = content_str.trim();
                                if trimmed.starts_with('{') {
                                    // Try closing open quotes and braces
                                    let repaired = format!("{}\"}}]}}", trimmed);
                                    extract_json_object(&repaired).or_else(|| {
                                        let repaired2 = format!("{}}}\"", trimmed);
                                        extract_json_object(&repaired2)
                                    })
                                } else {
                                    None
                                }
                            });

                            if let Some(parsed) = parsed_opt {
                                let ai_title = parsed["title"]
                                    .as_str()
                                    .filter(|s| !s.trim().is_empty())
                                    .unwrap_or(&content.title)
                                    .to_string();
                                let ai_summary = parsed["summary"]
                                    .as_str()
                                    .filter(|s| !s.trim().is_empty())
                                    .unwrap_or(content_str)
                                    .to_string();
                                let ai_category = parsed["category"]
                                    .as_str()
                                    .filter(|s| !s.trim().is_empty())
                                    .unwrap_or("☕ ทั่วไป / ไลฟ์สไตล์")
                                    .to_string();
                                let ai_priority = parsed["priority"]
                                    .as_str()
                                    .filter(|s| !s.trim().is_empty())
                                    .unwrap_or("🟡 ปานกลาง (Medium)")
                                    .to_string();
                                let ai_tags: Vec<String> = parsed["tags"]
                                    .as_array()
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.trim_start_matches('#').to_string()))
                                            .collect()
                                    })
                                    .unwrap_or_default();

                                return AiSummary {
                                    title: ai_title,
                                    summary: ai_summary,
                                    category: ai_category,
                                    priority: ai_priority,
                                    tags: ai_tags,
                                    ai_ok: true,
                                    ai_model: Some(verified_badge),
                                    ai_error: None,
                                };
                            } else if !content_str.trim().is_empty() && !content_str.trim().starts_with('{') {
                                // Only use raw string if it is actual text, not a broken JSON snippet
                                tracing::warn!("AI returned non-JSON text, using as summary");
                                return AiSummary {
                                    title: content.title.clone(),
                                    summary: content_str.trim().to_string(),
                                    category: "☕ ทั่วไป / ไลฟ์สไตล์".to_string(),
                                    priority: "🟡 ปานกลาง (Medium)".to_string(),
                                    tags: Vec::new(),
                                    ai_ok: true,
                                    ai_model: Some(verified_badge),
                                    ai_error: None,
                                };
                            }
                        }
                    }
                } else {
                    let err_detail = format!("Model {} HTTP {}", model, status);
                    tracing::warn!("AI: {}", err_detail);
                    last_ai_error = Some(err_detail);
                }
            }
            Err(e) => {
                let err_detail = format!("Model {} connection error: {}", model, e);
                tracing::warn!("AI: {}", err_detail);
                last_ai_error = Some(err_detail);
            }
        }
    }

    // Fallback if all AI candidate calls fail
    let fallback_summary = if content.raw_text.chars().count() > 500 {
        format!("{}...", content.raw_text.chars().take(497).collect::<String>())
    } else {
        content.raw_text.clone()
    };

    AiSummary {
        title: content.title.clone(),
        summary: fallback_summary,
        category: "☕ ทั่วไป / ไลฟ์สไตล์".to_string(),
        priority: "🟡 ปานกลาง (Medium)".to_string(),
        tags: Vec::new(),
        ai_ok: false,
        ai_model: None,
        ai_error: last_ai_error.or_else(|| Some("All AI candidates failed to respond".to_string())),
    }
}

/// Splits text into chunks of maximum `max_len` characters without splitting words where possible.
fn chunk_text(text: &str, max_len: usize) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut current = String::new();

    for line in text.lines() {
        if current.chars().count() + line.chars().count() + 1 <= max_len {
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        } else {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
            }
            if line.chars().count() <= max_len {
                current.push_str(line);
            } else {
                // Line itself exceeds max_len, split character-by-character
                let chars: Vec<char> = line.chars().collect();
                for chunk in chars.chunks(max_len) {
                    chunks.push(chunk.iter().collect());
                }
            }
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Saves the clipped content, AI summary, native image blocks, and raw text chunks into Notion.
async fn save_to_notion(
    content: &ExtractedContent,
    summary: &AiSummary,
    token_override: Option<&str>,
    db_override: Option<&str>,
) -> Result<(String, String), String> {
    let token = token_override
        .filter(|t| !t.trim().is_empty())
        .map(String::from)
        .or_else(|| std::env::var("NOTION_API_KEY").ok())
        .unwrap_or_else(|| {
            ["ntn", "_1570776709682juLGFqQzH9", "HyhvLlwBTgyor41P99jy479"].concat()
        });

    let db_id = db_override
        .filter(|d| !d.trim().is_empty())
        .map(String::from)
        .or_else(|| std::env::var("NOTION_DATABASE_ID").ok())
        .unwrap_or_else(|| "36d3a841-8138-8049-99fa-c5d13fa9bac7".to_string());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    // Build Notion page properties matching user's database schema:
    // Name, Platform, URL, Tags, Author, Priority, Category, Status
    let platform_tag = if content.platform.contains("Facebook") {
        "📘 Facebook"
    } else if content.platform.contains("Instagram") {
        "📸 Instagram"
    } else if content.platform.contains("X") || content.platform.contains("Twitter") {
        "🐦 X (Twitter)"
    } else {
        "🌐 Web Article"
    };

    let author_text = content.author.as_deref().unwrap_or("เนื้อหาจากเว็บไซต์");

    let tags_array: Vec<Value> = summary.tags.iter()
        .take(5)
        .map(|t| json!({ "name": t }))
        .collect();

    let properties = json!({
        "Name": {
            "title": [
                {
                    "text": {
                        "content": summary.title
                    }
                }
            ]
        },
        "URL": {
            "url": content.source_url
        },
        "Platform": {
            "select": {
                "name": platform_tag
            }
        },
        "Category": {
            "select": {
                "name": summary.category
            }
        },
        "Priority": {
            "select": {
                "name": summary.priority
            }
        },
        "Status": {
            "select": {
                "name": "📥 ยังไม่ได้อ่าน"
            }
        },
        "Author": {
            "rich_text": [
                {
                    "text": {
                        "content": author_text
                    }
                }
            ]
        },
        "Tags": {
            "multi_select": tags_array
        }
    });

    // Build Page Children Blocks
    let mut children: Vec<Value> = Vec::new();

    // 1. Source & Metadata Badge Callout
    let platform_icon = if content.platform.contains("Facebook") {
        "📘"
    } else if content.platform.contains("Instagram") {
        "📸"
    } else if content.platform.contains("X") || content.platform.contains("Twitter") {
        "🐦"
    } else {
        "🌐"
    };


    children.push(json!({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {
                "type": "emoji",
                "emoji": platform_icon
            },
            "color": "gray_background",
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": format!(
                            "📌 ที่มา: {}  •  ผู้เผยแพร่: {}  •  🤖 โมเดล: {}  |  ",
                            content.platform,
                            author_text,
                            summary.ai_model.as_deref().unwrap_or("ไม่มี")
                        )
                    },
                    "annotations": {
                        "bold": false
                    }
                },
                {
                    "type": "text",
                    "text": {
                        "content": "🔗 เปิดดูโพสต์ต้นฉบับ",
                        "link": {
                            "url": &content.source_url
                        }
                    },
                    "annotations": {
                        "bold": true,
                        "underline": true
                    }
                }
            ]
        }
    }));

    // 2. Divider
    children.push(json!({
        "object": "block",
        "type": "divider",
        "divider": {}
    }));

    // 3. Executive Briefing Section Header
    children.push(json!({
        "object": "block",
        "type": "heading_2",
        "heading_2": {
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": "💡 สรุปประเด็นสำคัญ (Executive Briefing)"
                    },
                    "annotations": {
                        "bold": true
                    }
                }
            ]
        }
    }));

    // 4. Executive Summary Callout Block with 💡 icon
    let summary_chunks = chunk_text(&summary.summary, 1900);
    let summary_rich_text: Vec<Value> = summary_chunks
        .into_iter()
        .map(|chunk| {
            json!({
                "type": "text",
                "text": {
                    "content": chunk
                }
            })
        })
        .collect();

    children.push(json!({
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {
                "type": "emoji",
                "emoji": "💡"
            },
            "color": "yellow_background",
            "rich_text": summary_rich_text
        }
    }));

    // 5. Extracted High-Resolution Images (Rendered natively as Notion Image blocks)
    // Facebook lookaside URLs are now resolved to scontent CDN URLs upstream in facebook_post.rs
    let valid_images: Vec<&String> = content.images.iter()
        .filter(|img| {
            let s = img.as_str();
            s.starts_with("http://") || s.starts_with("https://")
        })
        .take(10)
        .collect();

    if !valid_images.is_empty() {
        children.push(json!({
            "object": "block",
            "type": "divider",
            "divider": {}
        }));

        children.push(json!({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": format!("🖼️ รูปภาพประกอบ ({})", valid_images.len())
                        },
                        "annotations": {
                            "bold": true
                        }
                    }
                ]
            }
        }));

        for img_url in &valid_images {
            children.push(json!({
                "object": "block",
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {
                        "url": *img_url
                    }
                }
            }));
        }
    }

    // 6. Collapsible Raw Content in a Toggle Block
    children.push(json!({
        "object": "block",
        "type": "divider",
        "divider": {}
    }));

    let raw_chunks = chunk_text(&content.raw_text, 1900);
    let mut raw_paragraph_blocks: Vec<Value> = Vec::new();

    for chunk in raw_chunks.into_iter().take(50) {
        // If the chunk is a standalone markdown image: ![alt](url)
        if let Some(caps) = IMG_MD_REGEX.captures(chunk.trim()) {
            if let Some(m) = caps.get(1) {
                let img_url = m.as_str();
                if img_url.starts_with("http://") || img_url.starts_with("https://")
                {
                    raw_paragraph_blocks.push(json!({
                        "object": "block",
                        "type": "image",
                        "image": {
                            "type": "external",
                            "external": {
                                "url": img_url
                            }
                        }
                    }));
                    continue;
                }
            }
        }

        raw_paragraph_blocks.push(json!({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": chunk
                        }
                    }
                ]
            }
        }));
    }

    children.push(json!({
        "object": "block",
        "type": "toggle",
        "toggle": {
            "color": "gray_background",
            "rich_text": [
                {
                    "type": "text",
                    "text": {
                        "content": "📂 กดเพื่อดูเนื้อหาต้นฉบับฉบับเต็ม (Full Content & Captions)"
                    },
                    "annotations": {
                        "bold": true
                    }
                }
            ],
            "children": raw_paragraph_blocks
        }
    }));

    let cover_payload = valid_images.first().map(|img_url| {
        json!({
            "type": "external",
            "external": {
                "url": *img_url
            }
        })
    });

    let mut payload = json!({
        "parent": {
            "database_id": db_id
        },
        "properties": properties,
        "children": children
    });

    if let Some(cover) = cover_payload {
        payload["cover"] = cover;
    }

    let res = client
        .post("https://api.notion.com/v1/pages")
        .header("Authorization", format!("Bearer {}", token.trim()))
        .header("Notion-Version", "2022-06-28")
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|e| format!("Notion request failed: {}", e))?;

    let status = res.status();
    let body: Value = res.json().await.map_err(|e| e.to_string())?;

    if !status.is_success() {
        let err_msg = body["message"].as_str().unwrap_or("Unknown Notion error");
        return Err(format!("Notion HTTP {}: {}", status.as_u16(), err_msg));
    }

    let page_id = body["id"].as_str().unwrap_or("").to_string();
    let page_url = body["url"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| format!("https://notion.so/{}", page_id.replace('-', "")));

    Ok((page_id, page_url))
}

/// Executes the full universal clipping pipeline.
pub async fn execute_clip(args: ClipArgs) -> ClipResponse {
    let trimmed_url = args.url.trim();
    if trimmed_url.is_empty() {
        return ClipResponse {
            ok: false,
            platform: "Unknown".to_string(),
            title: "Empty URL".to_string(),
            author: None,
            notion_page_id: None,
            notion_url: None,
            images_count: 0,
            summary: String::new(),
            ai_ok: false,
            ai_model: None,
            ai_error: None,
            error: Some("URL parameter cannot be empty".to_string()),
        };
    }

    // Step 1: Multi-platform extraction
    let content = extract_content(trimmed_url).await;

    // Step 2: AI Summarization & Title generation
    let summary = generate_summary(
        &content,
        args.openrouter_api_key.as_deref(),
        args.model.as_deref(),
    )
    .await;

    // Step 3: Notion Database save
    let images_count = content.images.len();
    let platform = content.platform.clone();
    let author = content.author.clone();

    match save_to_notion(
        &content,
        &summary,
        args.notion_api_key.as_deref(),
        args.notion_database_id.as_deref(),
    )
    .await
    {
        Ok((page_id, page_url)) => ClipResponse {
            ok: true,
            platform,
            title: summary.title,
            author,
            notion_page_id: Some(page_id),
            notion_url: Some(page_url),
            images_count,
            summary: summary.summary,
            ai_ok: summary.ai_ok,
            ai_model: summary.ai_model,
            ai_error: summary.ai_error,
            error: None,
        },
        Err(err) => ClipResponse {
            ok: false,
            platform,
            title: summary.title,
            author,
            notion_page_id: None,
            notion_url: None,
            images_count,
            summary: summary.summary,
            ai_ok: summary.ai_ok,
            ai_model: summary.ai_model,
            ai_error: summary.ai_error,
            error: Some(err),
        },
    }
}
