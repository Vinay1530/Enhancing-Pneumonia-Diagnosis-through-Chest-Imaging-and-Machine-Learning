function submitImage() {
  // Get the image file input element
  var fileInput = document.getElementById('file-upload');

  // Check if a file is selected
  if (fileInput.files.length > 0) {
    // Get the selected file
    var file = fileInput.files[0];

    // Create a FormData object and append the file to it
    var formData = new FormData();
    formData.append('file', file);

    // Make a POST request to the Flask server
    fetch('/predict', {
      method: 'POST',
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      // Handle the response from the server
      console.log(data.result);

      console.log(data.raw_result);

      // Update the result element in the HTML
      var predResult = document.getElementById('pred-result');
      predResult.innerText = "Prediction: " + data.result;
      predResult.classList.remove('hidden');

      var imageDisplay = document.getElementById('image-display');
      imageDisplay.src = URL.createObjectURL(file);
      imageDisplay.classList.remove('hidden');

      // Display raw result
      var rawResultDiv = document.getElementById('raw-result');
      rawResultDiv.innerText = "Percentage: " + data.raw_result;
      rawResultDiv.classList.remove('hidden');
    })
    .catch(error => {
      console.error('Error:', error);
    });
  } else {
    console.error('No file selected.');
  }
}

function clearImage() {
  // Get the image display element
  var imageDisplay = document.getElementById('image-display');
  
  // Hide the image display
  imageDisplay.src = '';
  imageDisplay.classList.add('hidden');

  // Get the result element in the HTML
  var predResult = document.getElementById('pred-result');

  // Clear the result element text and hide it
  predResult.innerText = '';
  predResult.classList.add('hidden');

  // Get the file input element and reset its value
  var fileInput = document.getElementById('file-upload');
  fileInput.value = '';

  // Get the result element in the HTML
  var rawResultDiv = document.getElementById('raw-result');

  // Clear the result element text and hide it
  rawResultDiv.innerText = '';
  rawResultDiv.classList.add('hidden');

}

