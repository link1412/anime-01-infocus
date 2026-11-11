#version 410
// Final world image: lens glare over the integrated radiance. During a whip the samples are
// smeared along the pan so the shutter integrates a continuous path, not a row of ghosts.
in vec2 uv; out vec4 o;
uniform sampler2D uHdr,uGlare,uTight;
uniform vec2 uPx;
uniform float uBlur;
void main(){
  vec3 c;
  if(uBlur>.5){
    c=vec3(0.);
    const int N=9;
    for(int i=0;i<N;i++){
      float a=(float(i)+.5)/float(N)-.5;
      c+=texture(uHdr,uv+vec2(a*uBlur*uPx.x,0.)).rgb;
    }
    c/=float(N);
  } else c=texture(uHdr,uv).rgb;
  vec3 g=texture(uGlare,uv).rgb;
  vec3 t=texture(uTight,uv).rgb;
  // core + tight halo + long veil
  vec3 r=c*.975+t*.012+g*.013;
  o=vec4(r,1.);
}
