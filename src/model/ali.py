from torch import nn

from model.decoder import BaseDecoder
from model.discriminator import BaseDiscriminator
from model.encoder import BaseEncoder


class ALI(nn.Module):
    def __init__(
        self,
        enc: BaseEncoder,
        dec: BaseDecoder,
        dis: BaseDiscriminator,
    ):
        super().__init__()
        self.enc, self.dec, self.dis = enc, dec, dis

        assert self.enc.latent_dim == self.dec.latent_dim == self.dis.latent_dim
        self.latent_dim = self.enc.latent_dim

        self.enc.apply(self._weights_init)
        self.dec.apply(self._weights_init)
        self.dis.apply(self._weights_init)

    def _weights_init(self, m: nn.Module) -> None:
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.01)
            nn.init.constant_(m.bias.data, 0.0)
