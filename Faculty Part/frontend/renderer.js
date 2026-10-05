/**
 * Structured response renderer.
 * Uses document.createElement + textContent throughout — no innerHTML, no XSS.
 * A1: this file must be loaded before chat.js in chat.html.
 */

class ResponseRenderer {
    render(structured, sources = []) {
        if (structured.confidence === "none" || structured.fallback) {
            return this.renderFallback(structured.fallback);
        }

        const container = document.createElement("div");
        container.className = "response-container";

        if (structured.title) {
            const h = document.createElement("h2");
            h.className = "response-title";
            h.textContent = structured.title;
            container.appendChild(h);
        }

        if (structured.subtitle) {
            const p = document.createElement("p");
            p.className = "response-subtitle";
            p.textContent = structured.subtitle;
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
        
        // Parse footer text to identify source names and add collection tags
        // Footer format: "Based on: Source1, Source2, Source3"
        if (footerText.startsWith("Based on:")) {
            const prefix = "Based on: ";
            p.appendChild(document.createTextNode(prefix));
            
            const sourcesText = footerText.substring(prefix.length);
            const sourceNames = sourcesText.split(",").map(s => s.trim());
            
            sourceNames.forEach((sourceName, index) => {
                // Find matching source in sources array to get collection info
                const sourceInfo = sources.find(s => 
                    s.title === sourceName || 
                    s.document_name === sourceName ||
                    sourceName.includes(s.title) ||
                    sourceName.includes(s.document_name)
                );
                
                // Add source name
                p.appendChild(document.createTextNode(sourceName));
                
                // Add collection tag if available
                if (sourceInfo && sourceInfo.source_collection) {
                    const tag = document.createElement("span");
                    tag.className = `source-collection-tag source-collection-tag--${sourceInfo.source_collection}`;
                    tag.textContent = sourceInfo.source_collection;
                    p.appendChild(tag);
                }
                
                // Add comma separator if not last item
                if (index < sourceNames.length - 1) {
                    p.appendChild(document.createTextNode(", "));
                }
            });
        } else {
            // Fallback: just render as text
            p.textContent = footerText;
        }
        
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
            h.textContent = section.heading;
            div.appendChild(h);
        }
        const p = document.createElement("p");
        const text = section.content || "";
        text.split("\n").forEach((line, i, arr) => {
            p.appendChild(document.createTextNode(line));
            if (i < arr.length - 1) p.appendChild(document.createElement("br"));
        });
        div.appendChild(p);
        return div;
    }

    renderBullets(section) {
        const div = document.createElement("div");
        div.className = "response-section section-bullets";
        if (section.heading) {
            const h = document.createElement("h3");
            h.className = "section-heading";
            h.textContent = section.heading;
            div.appendChild(h);
        }
        const ul = document.createElement("ul");
        (section.items || []).forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
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
            h.textContent = section.heading;
            div.appendChild(h);
        }
        const ol = document.createElement("ol");
        (section.items || []).forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
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
            h.textContent = section.heading;
            div.appendChild(h);
        }
        
        if (!section.headers || !section.headers.length || !section.rows || !section.rows.length) {
            const p = document.createElement("p");
            p.textContent = "(Empty table)";
            p.style.color = "#6b7280";
            p.style.fontStyle = "italic";
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
                td.textContent = cell ?? "";
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
            h.textContent = section.heading;
            div.appendChild(h);
        }
        const p = document.createElement("p");
        p.textContent = section.content || "";
        div.appendChild(p);
        return div;
    }

    renderUnknown(section) {
        const div = document.createElement("div");
        div.className = "response-section section-unknown";
        const pre = document.createElement("pre");
        pre.style.cssText = "background:#1a1a1a;padding:1rem;border-radius:8px;overflow-x:auto;font-size:0.8rem;color:#9ca3af;";
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
