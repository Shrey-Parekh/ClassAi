/**
 * Structured response renderer.
 * Uses document.createElement + textContent throughout (no innerHTML, no XSS).
 * Loaded before chat.js in chat.html, and by index.html for the hero example.
 */

class ResponseRenderer {
    /**
     * Append text to el, turning the model's **bold** markdown into <strong>.
     * Text is only ever set through text nodes, so this stays XSS-safe.
     */
    inline(el, text) {
        String(text ?? "").split(/\*\*(.+?)\*\*/g).forEach((part, i) => {
            if (!part) return;
            if (i % 2) {
                const b = document.createElement("strong");
                b.textContent = part;
                el.appendChild(b);
            } else {
                el.appendChild(document.createTextNode(part));
            }
        });
        return el;
    }

    plain(text) {
        return String(text ?? "").replace(/\*\*(.+?)\*\*/g, "$1");
    }

    render(structured, sources = []) {
        if (structured.confidence === "none" || structured.fallback) {
            return this.renderFallback(structured.fallback);
        }

        const container = document.createElement("div");
        container.className = "response-container";

        if (structured.title) {
            const h = document.createElement("h2");
            h.className = "response-title";
            h.textContent = this.plain(structured.title);
            container.appendChild(h);
        }

        if (structured.subtitle) {
            const p = document.createElement("p");
            p.className = "response-subtitle";
            this.inline(p, structured.subtitle);
            container.appendChild(p);
        }

        (structured.sections || []).forEach(section => {
            container.appendChild(this.renderSection(section));
        });

        if (structured.footer) {
            container.appendChild(this.renderFooter(structured.footer, sources));
        }

        return container;
    }
    
    renderFooter(footerText, sources = []) {
        const p = document.createElement("p");
        p.className = "response-footer";

        // Footer format from the API: "Based on: Source1, Source2"
        if (!footerText.startsWith("Based on:")) {
            p.textContent = footerText;
            return p;
        }

        const label = document.createElement("span");
        label.className = "sources-label";
        label.textContent = "Sources";
        p.appendChild(label);

        footerText.substring("Based on:".length).split(",").map(s => s.trim()).filter(Boolean)
            .forEach(sourceName => {
                const info = sources.find(s =>
                    (s.title && (s.title === sourceName || sourceName.includes(s.title))) ||
                    (s.document_name && (s.document_name === sourceName || sourceName.includes(s.document_name)))
                );
                const chip = document.createElement("span");
                chip.className = "ref";
                chip.textContent = sourceName;
                p.appendChild(chip);

                if (info && info.source_collection) {
                    const tag = document.createElement("span");
                    tag.className = `source-collection-tag source-collection-tag--${info.source_collection}`;
                    tag.textContent = info.source_collection;
                    p.appendChild(tag);
                }
            });
        return p;
    }

    renderSection(section) {
        switch (section.type) {
            case "paragraph": return this.renderParagraph(section);
            case "bullets":   return this.renderBullets(section);
            case "steps":     return this.renderSteps(section);
            case "table":     return this.renderTable(section);
            case "alert":     return this.renderAlert(section);
            default: return this.renderUnknown(section);
        }
    }

    renderParagraph(section) {
        const div = document.createElement("div");
        div.className = "response-section section-paragraph";
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = this.plain(section.heading);
            div.appendChild(h);
        }
        // The model sometimes writes "- item" lines inside a paragraph; show
        // those as a real list and keep the remaining lines as text.
        const bullet = /^\s*(?:[-•*]|\d+[.)])\s+/;
        let p = null, ul = null;
        (section.content || "").split("\n").forEach(line => {
            if (!line.trim()) { p = null; ul = null; return; }
            if (bullet.test(line)) {
                if (!ul) { ul = document.createElement("ul"); div.appendChild(ul); div.classList.add("section-bullets"); }
                ul.appendChild(this.inline(document.createElement("li"), line.replace(bullet, "")));
                p = null;
            } else {
                if (!p) { p = document.createElement("p"); div.appendChild(p); }
                else p.appendChild(document.createElement("br"));
                this.inline(p, line);
                ul = null;
            }
        });
        return div;
    }

    renderBullets(section) {
        const div = document.createElement("div");
        div.className = "response-section section-bullets";
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = this.plain(section.heading);
            div.appendChild(h);
        }
        const ul = document.createElement("ul");
        (section.items || []).forEach(item => {
            const li = document.createElement("li");
            this.inline(li, item);
            ul.appendChild(li);
        });
        div.appendChild(ul);
        return div;
    }

    renderSteps(section) {
        const div = document.createElement("div");
        div.className = "response-section section-steps";
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = this.plain(section.heading);
            div.appendChild(h);
        }
        const ol = document.createElement("ol");
        (section.items || []).forEach(item => {
            const li = document.createElement("li");
            this.inline(li, item);
            ol.appendChild(li);
        });
        div.appendChild(ol);
        return div;
    }

    renderTable(section) {
        const div = document.createElement("div");
        div.className = "response-section section-table";
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = this.plain(section.heading);
            div.appendChild(h);
        }
        
        if (!section.headers || !section.headers.length || !section.rows || !section.rows.length) {
            const p = document.createElement("p");
            p.className = "table-empty";
            p.textContent = "No rows to show.";
            div.appendChild(p);
            return div;
        }

        const table = document.createElement("table");

        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        section.headers.forEach(header => {
            const th = document.createElement("th");
            th.textContent = header;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        const tbody = document.createElement("tbody");
        section.rows.forEach(row => {
            const tr = document.createElement("tr");
            row.forEach(cell => {
                const td = document.createElement("td");
                this.inline(td, cell);
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        div.appendChild(table);
        return div;
    }

    renderAlert(section) {
        const div = document.createElement("div");
        const sev = section.severity || "info";
        div.className = `response-section section-alert alert-${sev}`;
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = this.plain(section.heading);
            div.appendChild(h);
        }
        const p = document.createElement("p");
        this.inline(p, section.content);
        div.appendChild(p);
        return div;
    }

    renderUnknown(section) {
        const div = document.createElement("div");
        div.className = "response-section section-unknown";
        const pre = document.createElement("pre");
        pre.textContent = JSON.stringify(section, null, 2);
        div.appendChild(pre);
        return div;
    }

    renderFallback(message) {
        const div = document.createElement("div");
        div.className = "response-fallback";
        div.textContent = message || "I don't have that information in my current documents.";
        return div;
    }
}

const renderer = new ResponseRenderer();
