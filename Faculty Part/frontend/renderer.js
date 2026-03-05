/**
 * Structured response renderer for JSON-based LLM outputs.
 * 
 * Renders each section type with dedicated formatting.
 */

class ResponseRenderer {
    /**
     * Render a complete structured response.
     * @param {Object} structured - StructuredResponse object
     * @returns {HTMLElement} Rendered container
     */
    render(structured) {
        // Handle fallback
        if (structured.confidence === "none" || structured.fallback) {
            return this.renderFallback(structured.fallback);
        }

        const container = document.createElement("div");
        container.className = "response-container";

        // Title
        if (structured.title) {
            const title = document.createElement("h2");
            title.className = "response-title";
            title.textContent = structured.title;
            container.appendChild(title);
        }

        // Subtitle
        if (structured.subtitle) {
            const subtitle = document.createElement("p");
            subtitle.className = "response-subtitle";
            subtitle.textContent = structured.subtitle;
            container.appendChild(subtitle);
        }

        // Sections
        structured.sections.forEach(section => {
            container.appendChild(this.renderSection(section));
        });

        // Footer
        if (structured.footer) {
            const footer = document.createElement("p");
            footer.className = "response-footer";
            footer.textContent = structured.footer;
            container.appendChild(footer);
        }

        return container;
    }

    /**
     * Render a section based on its type.
     */
    renderSection(section) {
        switch(section.type) {
            case "paragraph": return this.renderParagraph(section);
            case "bullets":   return this.renderBullets(section);
            case "steps":     return this.renderSteps(section);
            case "table":     return this.renderTable(section);
            case "alert":     return this.renderAlert(section);
            default:          return this.renderParagraph(section);
        }
    }

    /**
     * Render paragraph section.
     */
    renderParagraph(section) {
        const div = document.createElement("div");
        div.className = "section-paragraph";

        if (section.heading) {
            const h = document.createElement("h3");
            h.textContent = section.heading;
            div.appendChild(h);
        }

        const p = document.createElement("p");
        p.textContent = section.content;
        div.appendChild(p);
        
        return div;
    }

    /**
     * Render bullet list section.
     */
    renderBullets(section) {
        const div = document.createElement("div");
        div.className = "section-bullets";

        if (section.heading) {
            const h = document.createElement("h3");
            h.textContent = section.heading;
            div.appendChild(h);
        }

        const ul = document.createElement("ul");
        section.items.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            ul.appendChild(li);
        });
        div.appendChild(ul);
        
        return div;
    }

    /**
     * Render numbered steps section.
     */
    renderSteps(section) {
        const div = document.createElement("div");
        div.className = "section-steps";

        if (section.heading) {
            const h = document.createElement("h3");
            h.textContent = section.heading;
            div.appendChild(h);
        }

        const ol = document.createElement("ol");
        section.items.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            ol.appendChild(li);
        });
        div.appendChild(ol);
        
        return div;
    }

    /**
     * Render table section.
     */
    renderTable(section) {
        const div = document.createElement("div");
        div.className = "section-table";

        if (section.heading) {
            const h = document.createElement("h3");
            h.textContent = section.heading;
            div.appendChild(h);
        }

        const table = document.createElement("table");
        
        // Header
        const thead = document.createElement("thead");
        const headerRow = document.createElement("tr");
        section.headers.forEach(header => {
            const th = document.createElement("th");
            th.textContent = header;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Body
        const tbody = document.createElement("tbody");
        section.rows.forEach(row => {
            const tr = document.createElement("tr");
            row.forEach(cell => {
                const td = document.createElement("td");
                td.textContent = cell;
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });

        table.appendChild(thead);
        table.appendChild(tbody);
        div.appendChild(table);
        
        return div;
    }

    /**
     * Render alert/warning section.
     */
    renderAlert(section) {
        const div = document.createElement("div");
        div.className = `section-alert alert-${section.severity}`;

        if (section.heading) {
            const h = document.createElement("h3");
            h.textContent = section.heading;
            div.appendChild(h);
        }

        const p = document.createElement("p");
        p.textContent = section.content;
        div.appendChild(p);
        
        return div;
    }

    /**
     * Render fallback message.
     */
    renderFallback(message) {
        const div = document.createElement("div");
        div.className = "response-fallback";
        div.textContent = message || "I don't have that information in my current documents.";
        return div;
    }
}

// Export renderer instance
const renderer = new ResponseRenderer();
