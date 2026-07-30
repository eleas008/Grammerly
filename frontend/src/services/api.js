const API_BASE_URL = 'http://localhost:5000';

export async function processText(text, mode) {
  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      mode,
    }),
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.statusText}`);
  }

  const data = await response.json();
  return data.output || '';
}