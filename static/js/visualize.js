document.addEventListener('DOMContentLoaded', function() {
    // 获取任务数据
    fetch(`/task/${taskId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('无法获取任务数据');
            }
            return response.json();
        })
        .then(data => {
            if (data.status !== 'completed') {
                alert('任务尚未完成，请稍后查看');
                window.location.href = '/';
                return;
            }
            
            // 加载结果数据
            loadResults(data.results);
        })
        .catch(error => {
            console.error('错误:', error);
            alert('加载结果失败: ' + error.message);
        });
    
    function loadResults(results) {
        // 处理人声轨道
        const vocals = results.vocals;
        
        // 设置音频播放器
        const vocalsAudio = document.getElementById('vocals-audio');
        vocalsAudio.src = `/output/${vocals.audio}`;
        
        // 设置下载链接
        document.getElementById('vocals-audio-download').href = `/output/${vocals.audio}`;
        document.getElementById('vocals-midi-download').href = `/output/${vocals.midi}`;
        document.getElementById('vocals-sheet-download').href = `/output/${vocals.sheet}`;
        document.getElementById('vocals-numbered-download').href = `/output/${vocals.numbered}`;
        
        // 显示歌词
        const lyricsContainer = document.getElementById('lyrics');
        if (vocals.lyrics) {
            // 确保歌词以中文格式显示
            lyricsContainer.innerHTML = vocals.lyrics.replace(/\n/g, '<br>');
        } else {
            lyricsContainer.textContent = '未能识别歌词';
        }
        
        // 处理伴奏主旋律轨道
        const instrumentsContainer = document.getElementById('instruments-container');
        
        // 检查伴奏轨道是否存在且不为空
        if (results.accompaniment && Object.keys(results.accompaniment).length > 0) {
            // 创建伴奏卡片
            const instrumentCard = document.createElement('div');
            instrumentCard.className = 'card shadow-sm mb-4';
            
            // 卡片内容
            instrumentCard.innerHTML = `
                <div class="card-header">
                    <h5 class="card-title mb-0">伴奏主旋律轨道</h5>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">音频预览</label>
                                <audio controls class="w-100" src="/output/${results.accompaniment.audio}"></audio>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-3">
                                <label class="form-label">下载文件</label>
                                <div class="list-group">
                                    <a href="/output/${results.accompaniment.audio}" class="list-group-item list-group-item-action">
                                        <i class="bi bi-file-earmark-music"></i> 伴奏音频文件
                                    </a>
                                    <a href="/output/${results.accompaniment.midi}" class="list-group-item list-group-item-action">
                                        <i class="bi bi-file-earmark-music"></i> MIDI文件
                                    </a>
                                    <a href="/output/${results.accompaniment.sheet}" class="list-group-item list-group-item-action">
                                        <i class="bi bi-file-earmark-music"></i> 五线谱文件
                                    </a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            // 添加到容器
            instrumentsContainer.appendChild(instrumentCard);
        } else {
            // 如果没有伴奏轨道，显示提示信息
            instrumentsContainer.innerHTML = '<div class="alert alert-info">未生成伴奏主旋律轨道</div>';
        }
    }
    
    // 格式化乐器名称
    function formatInstrumentName(name) {
        const nameMap = {
            'drums': '鼓',
            'bass': '贝斯',
            'piano': '钢琴',
            'other': '其他乐器'
        };
        
        return nameMap[name] || name;
    }
});