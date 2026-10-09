// Reference images stay local until the user explicitly runs a search.
// Canvas previews do not depend on img-src allowing local blob/data URLs.
export async function drawLocalPreview(canvas, blob, isCurrent) {
  const image = await createImageBitmap(blob);
  try {
    if (!isCurrent()) return false;
    canvas.width = image.width; canvas.height = image.height;
    canvas.getContext('2d').drawImage(image, 0, 0);
    return true;
  } finally { image.close(); }
}

export async function prepareSearchImage(file, maxBytes) {
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || !file.size || file.size > 10 * 1024 * 1024) {
    throw new Error('Choose a nonempty JPEG, PNG or WebP of at most 10 MB.');
  }
  if (!Number.isInteger(maxBytes) || maxBytes < 1) throw new Error('Image search is unavailable. Refresh the library.');
  const image = await createImageBitmap(file, {imageOrientation: 'from-image'});
  try {
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, 1024 / Math.max(image.width, image.height));
    canvas.width = Math.max(1, Math.round(image.width * scale));
    canvas.height = Math.max(1, Math.round(image.height * scale));
    const context = canvas.getContext('2d');
    context.fillStyle = '#fff'; context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    for (const quality of [.85, .7, .5]) {
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
      if (!blob || blob.size > maxBytes) continue;
      const data = await new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]); reader.onerror = () => reject(new Error('This image could not be read.')); reader.readAsDataURL(blob);
      });
      return {data, blob, width: canvas.width, height: canvas.height};
    }
    throw new Error('This image is too detailed for search. Choose a smaller image.');
  } finally { image.close(); }
}
