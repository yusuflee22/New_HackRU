// script.js
// Handles form submission and fetches ranked predictions from the API.

document.getElementById('rank-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const year = document.getElementById('year-input').value;
  const h = document.getElementById('horizon-select').value;
  const top = document.getElementById('top-input').value;
  const resultContainer = document.getElementById('result-container');
  resultContainer.innerHTML = '<p>Loading...</p>';
  // Build query string.  Encode year and horizon to integers.
  const query = new URLSearchParams({ year: year, h: h, top: top }).toString();
  try {
    const response = await fetch(`/rank?${query}`);
    const data = await response.json();
    if (response.ok) {
      if (Array.isArray(data) && data.length > 0) {
        // Build HTML table.  Include additional feature columns if available
        let html = '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200">';
        html += '<thead class="bg-gray-100"><tr>';
        html += '<th class="px-4 py-2 text-left text-sm font-medium text-gray-600">City</th>';
        html += '<th class="px-4 py-2 text-left text-sm font-medium text-gray-600">Predicted CAGR</th>';
        html += '<th class="px-4 py-2 text-left text-sm font-medium text-gray-600">Price Level</th>';
        html += '<th class="px-4 py-2 text-left text-sm font-medium text-gray-600">3Y Momentum</th>';
        html += '</tr></thead><tbody class="bg-white divide-y divide-gray-100">';
        data.forEach((item) => {
          const perc = (item.predicted_cagr * 100).toFixed(2);
          const priceLevel = (item.features && item.features.price_level != null)
            ? Number(item.features.price_level).toLocaleString(undefined, { maximumFractionDigits: 0 })
            : 'N/A';
          const momentum = (item.features && item.features.price_momentum_3y != null)
            ? (item.features.price_momentum_3y * 100).toFixed(2) + '%'
            : 'N/A';
          html += '<tr>';
          html += `<td class="px-4 py-2 text-sm text-gray-900">${item.city}</td>`;
          html += `<td class="px-4 py-2 text-sm text-gray-900">${perc}%</td>`;
          html += `<td class="px-4 py-2 text-sm text-gray-900">${priceLevel}</td>`;
          html += `<td class="px-4 py-2 text-sm text-gray-900">${momentum}</td>`;
          html += '</tr>';
        });
        html += '</tbody></table></div>';
        resultContainer.innerHTML = html;
      } else {
        resultContainer.innerHTML = '<p>No results found for the selected year.</p>';
      }
    } else {
      // Display server error message
      resultContainer.innerHTML = `<p class="text-red-600">${data.detail || 'Error fetching rankings.'}</p>`;
    }
  } catch (err) {
    resultContainer.innerHTML = '<p class="text-red-600">Error fetching data. Ensure the API is running.</p>';
    console.error(err);
  }
});