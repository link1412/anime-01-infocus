#version 410
// the last of the lens: a few dozen copies of a far-out-of-focus edge are still copies
in vec2 uv; out vec4 o;
uniform sampler2D uTex;
uniform vec2 uDir;
uniform float uSigma;
void main(){
  vec4 s=texture(uTex,uv); float w=1.;
  int n=int(min(ceil(uSigma*2.6),48.));
  for(int i=1;i<=n;i++){
    float g=exp(-.5*float(i*i)/(uSigma*uSigma));
    s+=(texture(uTex,uv+uDir*float(i))+texture(uTex,uv-uDir*float(i)))*g;
    w+=2.*g;
  }
  o=s/w;
}
