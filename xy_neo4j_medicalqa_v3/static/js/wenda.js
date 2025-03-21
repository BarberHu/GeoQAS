initKnowledgeGraph: function(containerId, kgData) {
    const container = document.getElementById(containerId || 'knowledgeGraphPreview');
    if (!container) {
        console.warn('找不到知识图谱容器');
        return;
    }

    const chart = echarts.init(container);
    const option = {
        tooltip: {
            show: true,
            formatter: function(params) {
                if (params.dataType === 'node') {
                    return `
                        <div class="kg-tooltip-node">
                            <h4>${params.name}</h4>
                            <p>类型: ${params.data.category || '未知'}</p>
                            ${params.data.description ? `<p>${params.data.description}</p>` : ''}
                        </div>`;
                }
            }
        },
        series: [{
            emphasis: {
                focus: 'adjacency',
                itemStyle: {
                    shadowBlur: 20,
                    shadowColor: 'rgba(0, 0, 0, 0.5)',
                    borderWidth: 2,
                    borderColor: '#409EFF'
                },
                lineStyle: {
                    width: 5,
                    opacity: 0.9,
                    color: '#409EFF'
                },
                label: {
                    show: true,
                    fontWeight: 'bold',
                    fontSize: 14,
                    backgroundColor: 'rgba(255, 255, 255, 0.9)',
                    padding: [4, 8],
                    borderRadius: 4
                }
            }
        }]
    };

    chart.setOption(option);
}
