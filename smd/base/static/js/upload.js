document.getElementById('imageUploadForm').addEventListener('submit', function (e) {
    e.preventDefault(); // Prevent default form submission

    const formData = new FormData();
    const fileInput = document.getElementById('imageFile');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select an image!');
        return;
    }

    formData.append('image', file); // Append the image file
    formData.append('dummy_data', 'example_value'); // Add any additional data

    axios.post('/api/upload/', formData, {
        headers: {
            'X-CSRFToken': getCookie('csrftoken'), // Include CSRF token
            'Content-Type': 'multipart/form-data',
        }
    })
    .then(response => {
        console.log('Upload success:', response.data);
        alert('Image uploaded successfully!');
    })
    .catch(error => {
        console.error('Upload error:', error);
        alert('Image upload failed: ' + error.message);
    });
});

// Function to get the CSRF token
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
