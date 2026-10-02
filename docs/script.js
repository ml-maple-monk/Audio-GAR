const datasetFilter = document.getElementById('dataset-filter');
const gapGroups = Array.from(document.querySelectorAll('#gap-table tbody[data-dataset]'));

if (datasetFilter && gapGroups.length) {
  document.getElementById('gap-controls').hidden = false;
  const updateDataset = () => {
    let count = 0;
    for (const group of gapGroups) {
      group.hidden = datasetFilter.value !== 'all' && group.dataset.dataset !== datasetFilter.value;
      if (!group.hidden) count += group.rows.length;
    }
    document.getElementById('gap-count').textContent = `${count} model-dataset comparisons`;
  };
  datasetFilter.addEventListener('change', updateDataset);
  updateDataset();
}

const copyButton = document.getElementById('copy-bibtex');
const bibtexCode = document.getElementById('bibtex-code');
const copyStatus = document.getElementById('copy-status');

if (copyButton && bibtexCode && copyStatus) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    let copied = false;
    try {
      await navigator.clipboard.writeText(bibtexCode.textContent);
      copied = true;
    } catch (_) {
      const textArea = document.createElement('textarea');
      textArea.value = bibtexCode.textContent;
      textArea.setAttribute('readonly', '');
      textArea.style.position = 'fixed';
      textArea.style.opacity = '0';
      document.body.appendChild(textArea);
      try {
        textArea.select();
        copied = document.execCommand('copy');
      } catch (_) {
        copied = false;
      } finally {
        textArea.remove();
        copyButton.focus({ preventScroll: true });
      }
    }
    copyStatus.textContent = copied ? 'BibTeX copied.' : 'Select the citation above to copy manually.';
  });
}
