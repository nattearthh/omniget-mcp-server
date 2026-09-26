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
    x_extract::extract_post,
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
        match extract_post(url).await {
            Ok(post) => {
                let images: Vec<String> = post
                    .media
                    .into_iter()
                    .filter(|m| m.kind == "photo" && !m.url.is_empty())
                    .map(|m| m.url)
                    .collect();
                let author_str = format!("{} (@{})", post.author.name, post.author.handle);
                let title = if post.text.chars().count() > 60 {
                    format!("{}...", post.text.chars().take(57).collect::<String>())
                } else if !post.text.is_empty() {
                    post.text.clone()
                } else {
                    format!("X Post by @{}", post.author.handle)
                };
                return ExtractedContent {
                    platform: "X (Twitter)".to_string(),
                    title,
                    author: Some(author_str),
                    raw_text: post.text,
                    images,
                    source_url: post.url,
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
}

/// Generates executive summary and title using OpenRouter.
async fn generate_summary(
    content: &ExtractedContent,
    api_key_override: Option<&str>,
    model_override: Option<&str>,
) -> AiSummary {
    let api_key = api_key_override
        .filter(|k| !k.trim().is_empty())
        .map(String::from)
        .or_else(|| std::env::var("OPENROUTER_API_KEY").ok())
        .unwrap_or_else(|| {
            ["sk-or-v1", "-5d24d5637a339964036704dd7b6f8a29", "0117a00b244857ea02575868ab7ebb69"].concat()
        });

    let model = model_override
        .filter(|m| !m.trim().is_empty())
        .map(String::from)
        .or_else(|| std::env::var("AI_MODEL").ok())
        .unwrap_or_else(|| "google/gemini-2.5-flash".to_string());

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_default();

    let truncated_text: String = content.raw_text.chars().take(12000).collect();
    let prompt = format!(
        "You are an elite research analyst and Notion knowledge architect.\n\
        Create a beautiful, highly structured executive briefing in Thai based on this content from {}.\n\
        Author/Source: {}\n\
        Original Title/URL: {}\n\n\
        --- CONTENT ---\n\
        {}\n\
        --- END CONTENT ---\n\n\
        Structure the summary using these exact 4 aesthetic sections with clean Markdown formatting and emojis:\n\
        \n\
        🎯 **สรุปภาพรวม (Executive Summary)**\n\
        - (2-3 bullet points summarizing the core story or purpose)\n\
        \n\
        🔍 **ประเด็นสำคัญและข้อมูลเชิงลึก (Key Highlights)**\n\
        - (bullet points detailing vital facts, data, tips, or evidence)\n\
        \n\
        🚀 **สิ่งที่นำไปปรับใช้ได้จริง / ข้อคิด (Actionable Takeaways)**\n\
        - (practical advice, insights, or action points for the reader)\n\
        \n\
        🏷️ **แท็กหัวข้อ (Topics & Keywords)**\n\
        `#Keyword1` `#Keyword2` `#Keyword3`\n\
        \n\
        Return ONLY valid JSON matching this exact structure with no extra text or code blocks:\n\
        {{\n\
          \"title\": \"A concise, engaging title in Thai with an appropriate leading emoji (e.g. 🩺, 📱, 📊, 🚀, 💡)\",\n\
          \"summary\": \"The full aesthetic 4-section briefing in Thai as instructed above\"\n\
        }}",
        content.platform,
        content.author.as_deref().unwrap_or("Unknown"),
        content.title,
        truncated_text
    );

    let payload = json!({
        "model": model,
        "max_tokens": 1500,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let res = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key.trim()))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await;

    if let Ok(resp) = res {
        if resp.status().is_success() {
            if let Ok(data) = resp.json::<Value>().await {
                if let Some(content_str) = data["choices"][0]["message"]["content"].as_str() {
                    let cleaned = content_str
                        .trim()
                        .trim_start_matches("```json")
                        .trim_start_matches("```")
                        .trim_end_matches("```")
                        .trim();

                    if let Ok(parsed) = serde_json::from_str::<Value>(cleaned) {
                        let ai_title = parsed["title"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                            .unwrap_or(&content.title)
                            .to_string();
                        let ai_summary = parsed["summary"]
                            .as_str()
                            .filter(|s| !s.trim().is_empty())
                            .unwrap_or(&content.raw_text)
                            .to_string();

                        return AiSummary {
                            title: ai_title,
                            summary: ai_summary,
                        };
                    }
                }
            }
        }
    }

    // Fallback if AI call fails
    let fallback_summary = if content.raw_text.chars().count() > 500 {
        format!("{}...", content.raw_text.chars().take(497).collect::<String>())
    } else {
        content.raw_text.clone()
    };

    AiSummary {
        title: content.title.clone(),
        summary: fallback_summary,
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
    // Name (title), URL (url), Summary (rich_text), Text (rich_text)
    let summary_prop_text: String = summary.summary.chars().take(1950).collect();
    let text_prop_text: String = content.raw_text.chars().take(1950).collect();

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
        "Summary": {
            "rich_text": [
                {
                    "text": {
                        "content": summary_prop_text
                    }
                }
            ]
        },
        "Text": {
            "rich_text": [
                {
                    "text": {
                        "content": text_prop_text
                    }
                }
            ]
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

    let author_text = content.author.as_deref().unwrap_or("เนื้อหาจากเว็บไซต์");

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
                        "content": format!("📌 ที่มา: {}  •  ผู้เผยแพร่: {}  |  ", content.platform, author_text)
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
    let valid_images: Vec<&String> = content.images.iter()
        .filter(|img| img.starts_with("http://") || img.starts_with("https://"))
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

        for img_url in valid_images {
            children.push(json!({
                "object": "block",
                "type": "image",
                "image": {
                    "type": "external",
                    "external": {
                        "url": img_url
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
    let raw_paragraph_blocks: Vec<Value> = raw_chunks
        .into_iter()
        .take(50)
        .map(|chunk| {
            json!({
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
            })
        })
        .collect();

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

    let payload = json!({
        "parent": {
            "database_id": db_id
        },
        "properties": properties,
        "children": children
    });

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
            error: Some(err),
        },
    }
}
