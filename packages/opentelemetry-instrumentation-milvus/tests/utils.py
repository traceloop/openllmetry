def find_metrics_by_name(metrics_data, target_name):
    """Return a list of metrics with the given name from the reader."""
    matching_metrics = []
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == target_name:
                    matching_metrics.append(metric)
    return matching_metrics
