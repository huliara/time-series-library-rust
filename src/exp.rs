use crate::args::RootArgs as Args;

pub fn _get_exp_name(args: &Args) -> String {
    format!(
        "{:?}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}",
        args.task_name,
        args.model_id,
        args.model_config,
        args.data_config.data,
        args.data_config.feature_type,
        args.time_lengths.seq_len,
        args.time_lengths.label_len,
        args.time_lengths.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.data_config.embed,
        args.distil,
        args.des,
    )
}
