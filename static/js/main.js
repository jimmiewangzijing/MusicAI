document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const audioFileInput = document.getElementById('audio-file');
    const processingStatus = document.getElementById('processing-status');
    const progressBar = document.getElementById('progress-bar');
    const statusMessage = document.getElementById('status-message');
    const viewResults = document.getElementById('view-results');
    const resultsLink = document.getElementById('results-link');
    
    let currentTaskId = null;
    let statusCheckInterval = null;

    // 处理文件上传
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const file = audioFileInput.files[0];
        if (!file) {
            alert('请选择音频文件');
            return;
        }
        
        // 检查文件类型
        const allowedTypes = ['audio/mp3', 'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac'];
        if (!allowedTypes.includes(file.type) && 
            !file.name.endsWith('.mp3') && 
            !file.name.endsWith('.wav') && 
            !file.name.endsWith('.flac')) {
            alert('请选择MP3、WAV或FLAC格式的音频文件');
            return;
        }
        
        // 准备上传
        const formData = new FormData();
        formData.append('file', file);
        
        // 禁用上传按钮
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="bi bi-hourglass"></i> 上传中...';
        
        // 发送上传请求
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('上传失败');
            }
            return response.json();
        })
        .then(data => {
            // 显示处理状态
            processingStatus.classList.remove('d-none');
            currentTaskId = data.task_id;
            
            // 开始定期检查任务状态
            statusCheckInterval = setInterval(checkTaskStatus, 2000);
        })
        .catch(error => {
            console.error('错误:', error);
            alert('上传失败: ' + error.message);
            
            // 重置上传按钮
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="bi bi-cloud-upload"></i> 上传并处理';
        });
    });
    
    // 检查任务状态
    function checkTaskStatus() {
        if (!currentTaskId) return;
        
        fetch(`/task/${currentTaskId}`)
        .then(response => response.json())
        .then(data => {
            // 更新进度条和状态消息
            progressBar.style.width = `${data.progress}%`;
            statusMessage.textContent = data.message;
            
            // 处理完成或出错
            if (data.status === 'completed') {
                clearInterval(statusCheckInterval);
                statusMessage.textContent = '处理完成！';
                viewResults.classList.remove('d-none');
                resultsLink.href = `/visualize/${currentTaskId}`;
                
                // 重置上传按钮
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<i class="bi bi-cloud-upload"></i> 上传并处理';
            } else if (data.status === 'error') {
                clearInterval(statusCheckInterval);
                statusMessage.textContent = `处理失败: ${data.message}`;
                progressBar.classList.remove('progress-bar-animated');
                progressBar.classList.add('bg-danger');
                
                // 重置上传按钮
                uploadBtn.disabled = false;
                uploadBtn.innerHTML = '<i class="bi bi-cloud-upload"></i> 上传并处理';
            }
        })
        .catch(error => {
            console.error('状态检查错误:', error);
        });
    }
}); 