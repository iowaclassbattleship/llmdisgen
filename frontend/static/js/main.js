const renderPaper = (page, sections) => {
    for (const section of sections) {
        const h2 = document.createElement("h2");
        h2.innerText = section.header;
    }
}