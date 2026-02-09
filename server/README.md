# FlowEdit Server

A FastAPI-based REST API server for text-guided image editing using the FLUX model via FlowEdit.

## Features

- **Job-based workflow**: Create job → Submit edit → Poll status → Download result
- **Text-guided editing**: Edit images using source and target prompts
- **FLUX model**: Uses FLUX.1-dev for high-quality image editing
- **Async processing**: Long-running edits are executed asynchronously with status tracking
- **FastAPI framework**: Modern, fast, with automatic API documentation
- **Request validation**: Automatic validation of inputs using Pydantic
- **Error handling**: Comprehensive error handling with clear error messages

## Installation

1. Install the server dependencies:
```bash
pip install -r server/requirements.txt
```

2. Ensure FlowEdit dependencies are installed (torch, diffusers, transformers, etc.)

3. Ensure you have CUDA-capable GPU for best performance (CPU mode is supported but slow)

## Configuration

The server can be configured using environment variables:

- `FLOWEDIT_JOBS_DIR`: Directory for job data (default: `server/jobs/`)
- `FLOWEDIT_MODEL_PATH`: FLUX model path (default: `black-forest-labs/FLUX.1-dev`)
- `FLOWEDIT_DEVICE`: Device to use - `cuda` or `cpu` (default: auto-detect)
- `FLOWEDIT_MAX_CONCURRENT_JOBS`: Maximum parallel jobs (default: `1`)
- `PORT`: Server port (default: `8001`)
- `HOST`: Server host (default: `0.0.0.0`)

## Running the Server

### Basic usage:
From the project root directory:
```bash
python -m server.main
```

Or using uvicorn directly:
```bash
uvicorn server.main:app --host 0.0.0.0 --port 8001
```

### With custom configuration:
```bash
FLOWEDIT_DEVICE=cuda FLOWEDIT_MAX_CONCURRENT_JOBS=2 python -m server.main
```

The server will start on `http://localhost:8001`

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8001/docs
- **Alternative docs**: http://localhost:8001/redoc

## API Endpoints

### POST /jobs

Create a new edit job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "uploading",
  "message": "Job created successfully"
}
```

### POST /jobs/{job_id}/edit

Submit an edit request.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `image`: Image file (JPEG or PNG)
  - `source_prompt`: Source prompt describing the input image (string)
  - `target_prompt`: Target prompt describing desired edit (string)
  - `edit_params_json`: Optional JSON string with edit parameters:
    ```json
    {
      "T_steps": 28,
      "n_avg": 1,
      "src_guidance_scale": 1.5,
      "tar_guidance_scale": 5.5,
      "n_min": 0,
      "n_max": 24
    }
    ```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Edit submitted successfully"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8001/jobs/{job_id}/edit" \
  -F "image=@path/to/image.jpg" \
  -F "source_prompt=A cat sitting on a chair" \
  -F "target_prompt=A dog sitting on a chair" \
  -F 'edit_params_json={"T_steps": 28, "n_avg": 1, "src_guidance_scale": 1.5, "tar_guidance_scale": 5.5, "n_min": 0, "n_max": 24}'
```

### GET /jobs/{job_id}

Get job status and information.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "running",
  "created_at": "2024-01-15T14:30:45",
  "updated_at": "2024-01-15T14:31:20",
  "source_prompt": "A cat sitting on a chair",
  "target_prompt": "A dog sitting on a chair",
  "edit_params": {
    "T_steps": 28,
    "n_avg": 1,
    "src_guidance_scale": 1.5,
    "tar_guidance_scale": 5.5,
    "n_min": 0,
    "n_max": 24
  },
  "result_path": "jobs/{job_id}/output/edited_image.png",
  "error_message": null
}
```

**Job Status Values:**
- `uploading`: Job created, waiting for edit submission
- `pending`: Edit submitted, waiting to start processing
- `running`: Edit is currently being processed
- `completed`: Edit completed successfully
- `failed`: Edit failed with error
- `cancelled`: Job was cancelled

### GET /jobs/{job_id}/result

Download edited image result.

**Response:**
- Content-Type: `image/png`
- File download of edited image

**Example using curl:**
```bash
curl -X GET "http://localhost:8001/jobs/{job_id}/result" \
  --output edited_image.png
```

### DELETE /jobs/{job_id}

Cancel or delete a job.

**Response:**
```json
{
  "message": "Job {job_id} cancelled successfully"
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_jobs": 1,
  "total_jobs": 5
}
```

## Edit Parameters

Default parameters (from FLUX_exp.yaml):

- `T_steps`: 28 - Number of timesteps for diffusion
- `n_avg`: 1 - Number of averaging iterations
- `src_guidance_scale`: 1.5 - Source guidance scale
- `tar_guidance_scale`: 5.5 - Target guidance scale
- `n_min`: 0 - Minimum step for editing
- `n_max`: 24 - Maximum step for editing

These can be customized per request via the `edit_params_json` parameter.

## Job Lifecycle

1. **Create Job**: `POST /jobs` → Returns `job_id` with status `uploading`
2. **Submit Edit**: `POST /jobs/{job_id}/edit` → Upload image + prompts → Status becomes `pending`
3. **Poll Status**: `GET /jobs/{job_id}` → Check status (`pending` → `running` → `completed`/`failed`)
4. **Download Result**: `GET /jobs/{job_id}/result` → Download edited image file

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `201`: Created (job creation)
- `400`: Bad Request (invalid image format, invalid prompts)
- `404`: Not Found (job not found, result not found)
- `500`: Internal Server Error (processing errors)
- `503`: Service Unavailable (model not loaded, max concurrent jobs reached)

## Example Python Client

```python
import requests

# Create job
response = requests.post("http://localhost:8001/jobs")
job_data = response.json()
job_id = job_data["job_id"]

# Submit edit
with open("input_image.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "source_prompt": "A cat sitting on a chair",
        "target_prompt": "A dog sitting on a chair"
    }
    response = requests.post(
        f"http://localhost:8001/jobs/{job_id}/edit",
        files=files,
        data=data
    )

# Poll status
while True:
    response = requests.get(f"http://localhost:8001/jobs/{job_id}")
    status = response.json()["status"]
    
    if status == "completed":
        # Download result
        response = requests.get(f"http://localhost:8001/jobs/{job_id}/result")
        with open("edited_image.png", "wb") as f:
            f.write(response.content)
        break
    elif status == "failed":
        print(f"Edit failed: {response.json()['error_message']}")
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

## Example C++ Client Usage

```cpp
#include "rpc/FlowEditClient.h"

FlowEditClient client;
client.setEndpoint("http://localhost:8001");

// Create job
client.createJob([](const FlowEditClient::SubmitResult& result) {
    if (result.success) {
        std::string jobId = result.jobId;
        
        // Submit edit
        std::vector<uint8_t> imageData = loadImage("input.jpg");
        FlowEditClient::EditParams params;  // Uses defaults
        
        client.submitEdit(jobId, imageData,
                         "A cat sitting on a chair",
                         "A dog sitting on a chair",
                         params,
                         [](const FlowEditClient::SubmitResult& result) {
                             // Edit submitted
                         });
        
        // Poll status
        client.pollJobStatus(jobId, [](const FlowEditClient::JobInfo& info) {
            if (info.status == FlowEditClient::JobStatus::COMPLETED) {
                // Download result
                client.downloadResult(info.jobId, "output/",
                                    [](const FlowEditClient::DownloadResult& result) {
                                        // Result downloaded
                                    });
            }
        });
    }
});

// In main loop
client.frameUpdate();
```

## Troubleshooting

1. **Model not loading**: Ensure CUDA is available or set `FLOWEDIT_DEVICE=cpu`
2. **CUDA errors**: Check GPU availability and CUDA installation
3. **Memory errors**: Reduce image size or use CPU mode
4. **Import errors**: Ensure FlowEdit dependencies are installed in the same Python environment
5. **Port conflicts**: Change port using `PORT` environment variable

## Notes

- The model is loaded once at startup and kept in memory for all requests
- Large images may take longer to process
- GPU is recommended for better performance (edits can take 30-60 seconds on GPU, several minutes on CPU)
- Job directories are created under `jobs/` and contain input images, output images, and logs
- Completed jobs are kept for reference; use DELETE endpoint to clean up
