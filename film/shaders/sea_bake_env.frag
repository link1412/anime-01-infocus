#version 410
// The sky the water reflects: clear sky plus clouds, no sun. u azimuth, v linear elevation.
in vec2 uv; out vec4 o;
uniform sampler2D uBg;
uniform sampler2DArray uCloud;
uniform sampler2D uStacks;
uniform vec4 uStackWin;
void main(){
  float el01=uv.y;                 // elevation / (pi/2)
  vec2 p=vec2(uv.x,sqrt(el01));
  vec3 acc=vec3(0.); float T=1.;
  for(int k=0;k<3;k++){
    vec4 c=textureLod(uCloud,vec3(p,float(k)),0.);
    acc+=T*c.rgb; T*=c.a;
  }
  vec3 c=acc+T*texture(uBg,p).rgb;
  // the rock stands in the mirror too
  float az=(uv.x-.5)*6.2831853,el=el01*1.5707963;
  vec2 w=(vec2(az,el)-uStackWin.xz)/(uStackWin.yw-uStackWin.xz);
  if(w.x>0.&&w.x<1.&&w.y<1.){ vec4 s=textureLod(uStacks,w,2.); c=c*(1.-s.a)+s.rgb; }
  o=vec4(c,1.);
}
