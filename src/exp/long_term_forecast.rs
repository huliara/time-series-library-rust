use crate::exp::Exp;
use crate::{args::Args, exp::get_exp_name};
use burn::record::CompactRecorder;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::{
    backend::wgpu::Wgpu,
    module::AutodiffModule,
    optim::{adaptor::OptimizerAdaptor, decay::WeightDecayConfig, Adam, AdamConfig},
    tensor::backend::AutodiffBackend,
    train::LearnerBuilder,
};

use std::time::Instant;

type Backend = Wgpu;
pub struct ExpLongTermForecast {
    exp_name: String,
    args: Args,
}

impl ExpLongTermForecast {
    pub fn new(args: Args) -> Self {
        Self {
            exp_name: get_exp_name(&args),
            args,
        }
    }
    pub fn train<B: AutodiffBackend, M: AutodiffModule<B>>(&self, args: &Args) {
        let _train_data = (); // self._get_data(flag='train')
        let _vali_data = (); // self._get_data(flag='val')
        let _test_data = (); // self._get_data(flag='test')
        
        // TODO: Config and Model initialization
        /*
        let optimizer: OptimizerAdaptor<Adam, M, B> = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_weight_decay(Some(WeightDecayConfig::new(0.01)))
            .init();

        let device = B::Device::default();

        let leaner = LearnerBuilder::new(self.exp_name.clone())
            .metric_train_numeric(AccuracyMetric::new())
            .metric_valid_numeric(AccuracyMetric::new())
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(self.args.train_epochs)
            .summary()
            .build(
                config.model.init::<B>(&device),
                config.optimizer.init(),
                config.learning_rate,
            );
        */

        let path = format!("{}/{}", args.checkpoints, "setting_placeholder");
        if !std::path::Path::new(&path).exists() {
            std::fs::create_dir_all(&path).unwrap_or_else(|_| {});
        }

        let _time_now = Instant::now();
        // let early_stopping = EarlyStopping(patience=self.args.patience, verbose=True);

        let _model_optim = self._select_optimizer(args);
        let _criterion = self._select_criterion();

        if args.use_amp {
            // scaler = torch.cuda.amp.GradScaler()
        }

        // Loop epochs
        // for epoch in 0..self.args.train_epochs { ... }
    }
    
    fn _select_optimizer(&self, _args: &Args) {
        // TODO
    }
    
    fn _select_criterion(&self) {
        // TODO
    }

    fn validate(&self, _args: &Args) {
        // self.vali with validation data
    }

    fn test(&self, _args: &Args) {
        // Test logic
    }
}
