document.getElementById('predict-button').addEventListener('click', () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Update HTML with prediction result
            document.getElementById('predictionResult').innerHTML = `
                <p>Prediction: ${data.prediction}</p>
                <p>Confidence: ${data.confidence}</p>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Please select an image file.');
    }
});


// Get the result container element
const resultContainer = document.getElementById('resultContainer');

// Function to show the result container
function showResultContainer() {
  resultContainer.style.display = 'block'; // Show the container
}

// Add event listeners to the buttons to show the result container
document.getElementById('predict-button').addEventListener('click', showResultContainer);
// Add event listeners to other buttons if needed


// Function to handle file input change event
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();
    
    reader.onload = function(event) {
        const image = new Image();
        image.src = event.target.result;
        image.style.maxWidth = '200px'; // Adjust image preview size as needed
        image.style.marginTop = '10px'; // Adjust margin as needed
        
        const imageName = document.createElement('p');
        imageName.textContent = file.name;
        
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.innerHTML = ''; // Clear previous preview
        imagePreview.appendChild(image);
        imagePreview.appendChild(imageName);
    };
    
    reader.readAsDataURL(file);
});


// Get all question elements
const questions = document.querySelectorAll('.question');

// Add click event listener to each question
questions.forEach(question => {
    question.addEventListener('click', () => {
        // Toggle the visibility of the answer
        const answer = question.nextElementSibling;
        answer.classList.toggle('show-answer');

        // Change arrow direction
        const arrow = question.querySelector('.arrow');
        arrow.textContent = answer.classList.contains('show-answer') ? '▲' : '▼';
    });
});

