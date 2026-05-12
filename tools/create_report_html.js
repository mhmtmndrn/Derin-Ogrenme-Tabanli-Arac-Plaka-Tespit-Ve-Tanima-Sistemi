const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const inputPath = path.join(root, "docs", "rapor.md");
const outputPath = path.join(root, "docs", "rapor.html");
const markdown = fs.readFileSync(inputPath, "utf8");

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(value) {
  return escapeHtml(value).replace(/`([^`]+)`/g, "<code>$1</code>");
}

function closeLists(state, html) {
  if (state.ul) {
    html.push("</ul>");
    state.ul = false;
  }
  if (state.ol) {
    html.push("</ol>");
    state.ol = false;
  }
}

const html = [];
const state = { ul: false, ol: false, code: false, codeLines: [] };

for (const line of markdown.split(/\r?\n/)) {
  if (line.startsWith("```")) {
    if (state.code) {
      html.push(`<pre><code>${escapeHtml(state.codeLines.join("\n"))}</code></pre>`);
      state.code = false;
      state.codeLines = [];
    } else {
      closeLists(state, html);
      state.code = true;
    }
    continue;
  }

  if (state.code) {
    state.codeLines.push(line);
    continue;
  }

  if (!line.trim()) {
    closeLists(state, html);
    continue;
  }

  const imageMatch = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
  if (imageMatch) {
    closeLists(state, html);
    html.push(`<figure><img src="${escapeHtml(imageMatch[2])}" alt="${escapeHtml(imageMatch[1])}"><figcaption>${escapeHtml(imageMatch[1])}</figcaption></figure>`);
  } else if (line.startsWith("# ")) {
    closeLists(state, html);
    html.push(`<h1>${inlineMarkdown(line.slice(2).trim())}</h1>`);
  } else if (line.startsWith("## ")) {
    closeLists(state, html);
    html.push(`<h2>${inlineMarkdown(line.slice(3).trim())}</h2>`);
  } else if (line.startsWith("### ")) {
    closeLists(state, html);
    html.push(`<h3>${inlineMarkdown(line.slice(4).trim())}</h3>`);
  } else if (line.startsWith("- ")) {
    if (!state.ul) {
      closeLists(state, html);
      html.push("<ul>");
      state.ul = true;
    }
    html.push(`<li>${inlineMarkdown(line.slice(2).trim())}</li>`);
  } else if (/^\d+\.\s+/.test(line)) {
    if (!state.ol) {
      closeLists(state, html);
      html.push("<ol>");
      state.ol = true;
    }
    html.push(`<li>${inlineMarkdown(line.replace(/^\d+\.\s+/, "").trim())}</li>`);
  } else {
    closeLists(state, html);
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  }
}

closeLists(state, html);

fs.writeFileSync(
  outputPath,
  `<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Derin Öğrenme Tabanlı Araç Plaka Tespit ve Okuma Sistemi</title>
  <style>
    @page { margin: 22mm 18mm; }
    body { color: #1f2933; font-family: Arial, sans-serif; font-size: 11pt; line-height: 1.45; }
    h1 { color: #111827; font-size: 22pt; margin: 0 0 16px; }
    h2 { border-bottom: 1px solid #d0d7de; color: #111827; font-size: 15pt; margin: 22px 0 8px; padding-bottom: 4px; }
    h3 { font-size: 12.5pt; margin: 16px 0 6px; }
    p { margin: 7px 0; }
    ul, ol { margin: 7px 0 10px 22px; padding: 0; }
    li { margin: 3px 0; }
    code { background: #f2f4f7; border-radius: 3px; font-family: Consolas, monospace; font-size: 10pt; padding: 1px 4px; }
    pre { background: #f8fafc; border: 1px solid #d0d7de; border-radius: 6px; font-size: 9pt; overflow-wrap: anywhere; padding: 10px; white-space: pre-wrap; }
    figure { margin: 14px 0 18px; page-break-inside: avoid; text-align: center; }
    figure img { max-width: 100%; width: 6.7in; height: auto; }
    figcaption { color: #475569; font-size: 9.5pt; margin-top: 4px; }
  </style>
</head>
<body>
${html.join("\n")}
</body>
</html>
`,
  "utf8"
);

console.log(`Created ${outputPath}`);
