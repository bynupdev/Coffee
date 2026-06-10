document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImageBtn = document.getElementById('removeImage');
    const submitBtn = document.getElementById('submitBtn');
    const resultsSection = document.getElementById('results');
    const loadingSection = document.getElementById('loading');
    const errorMessage = document.getElementById('errorMessage');
    
    const gradeResult = document.getElementById('gradeResult');
    const confidenceResult = document.getElementById('confidenceResult');
    const confidenceBar = document.getElementById('confidenceBar');
    const allProbabilities = document.getElementById('allProbabilities');

    // Image preview
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
                errorMessage.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    // Remove image
    removeImageBtn.addEventListener('click', function() {
        imageInput.value = '';
        previewContainer.style.display = 'none';
        imagePreview.src = '';
    });

    // Form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = imageInput.files[0];
        if (!file) {
            showError('Please select an image file');
            return;
        }

        // Validate file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            showError('Please select a valid image file (JPEG or PNG)');
            return;
        }

        // Validate file size (10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('File size must be less than 10MB');
            return;
        }

        // Show loading
        showLoading();
        
        // Prepare form data
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/predict/', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': getCsrfToken()
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Prediction failed');
            }

            // Display results
            displayResults(data);
            
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading();
        }
    });

    function displayResults(data) {
        // Update main results
        gradeResult.textContent = data.grade;
        confidenceResult.textContent = `${(data.confidence * 100).toFixed(2)}%`;
        confidenceBar.style.width = `${data.confidence * 100}%`;
        
        // Display all probabilities
        if (data.details) {
            let html = '<h3>All Grades Probability:</h3>';
            for (const [grade, prob] of Object.entries(data.details)) {
                html += `
                    <div class="probability-item">
                        <span>Grade ${grade}:</span>
                        <span>${(prob * 100).toFixed(2)}%</span>
                    </div>
                `;
            }
            allProbabilities.innerHTML = html;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        errorMessage.style.display = 'none';
        
        // Smooth scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showLoading() {
        loadingSection.style.display = 'block';
        resultsSection.style.display = 'none';
        errorMessage.style.display = 'none';
        submitBtn.disabled = true;
    }

    function hideLoading() {
        loadingSection.style.display = 'none';
        submitBtn.disabled = false;
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        resultsSection.style.display = 'none';
        loadingSection.style.display = 'none';
        submitBtn.disabled = false;
    }

    function getCsrfToken() {
        const name = 'csrftoken';
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

    // Drag and drop support
    const uploadSection = document.querySelector('.upload-section');
    
    uploadSection.addEventListener('dragover', function(e) {
        e.preventDefault();
        this.style.border = '2px dashed var(--primary)';
    });

    uploadSection.addEventListener('dragleave', function(e) {
        e.preventDefault();
        this.style.border = 'none';
    });

    uploadSection.addEventListener('drop', function(e) {
        e.preventDefault();
        this.style.border = 'none';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            imageInput.files = e.dataTransfer.files;
            
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            imageInput.dispatchEvent(event);
        }
    });
});